#!/usr/bin/env python3
"""Response-only LoRA SFT for the Qwen3.5 video-safety Agent.

This intentionally uses a small custom loop: every record can contain a
different number of video frames, and only the assistant JSON is supervised.
The vision encoder and base language-model weights stay frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from agent.video import extract_frames, recover_frame_near_timestamp
from json_field_supervision import token_positions, validate_classification_context
from sparse_sft_loss import selected_targets, selected_loss, selected_subset_loss
from recorded_request_content import render_state_content
from recorded_frame_replay import recorded_frames
from trim_sft_suffix import trim_masked_suffix


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "Qwen3.5-4B"
DEFAULT_DATA = ROOT / "video_detection_agent/outputs/sft_coldstart_v2/train_examples.jsonl"
DEFAULT_OUT = ROOT / "video_detection_agent/outputs/sft_qwen35_4b_coldstart_v2"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_unified_options(args, rows):
    if not any(r.get('contract_version') == 'unified-student-next-step-v1' for r in rows):
        return
    if not all(r.get('contract_version') == 'unified-student-next-step-v1' for r in rows):
        raise ValueError('Do not mix legacy route/head objectives into unified next-step data')
    if args.route_only or args.supervise_json_fields or args.supervise_action_stage or args.route_field_only:
        raise ValueError('Unified Student needs complete approved outputs, not action-only masks')
    if args.adapter is not None:
        raise ValueError('This unified pilot starts a fresh LoRA from base; no old Adapter combination')
    if args.max_pixels != 409600:
        raise ValueError('Unified image contract requires --max-pixels=409600')


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_to_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))


class AgentSFTDataset(Dataset):
    def __init__(self, path: Path, max_examples: int = 0, longest_first: bool = False,
                 trajectory_group: str | None = None):
        self.rows = read_jsonl(path)
        if trajectory_group:
            self.rows = [
                row for row in self.rows
                if row.get("trajectory_group") == trajectory_group
            ]
        if longest_first:
            self.rows.sort(
                key=lambda row: (
                    row.get("image_count", len(row.get("timestamps") or []))
                    + sum(len(turn.get("text", "")) for turn in row.get("turns") or [])
                ),
                reverse=True,
            )
        if max_examples:
            self.rows = self.rows[:max_examples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ResponseOnlyCollator:
    def __init__(self, processor, max_pixels: int, max_sequence_length: int,
                 supervise_action_stage: str | None = None,
                 supervise_action_field: bool = False,
                 supervise_json_fields: tuple[str, ...] = (),
                 allow_json_integer_null: bool = False):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_sequence_length = max_sequence_length
        self.supervise_action_stage = supervise_action_stage
        self.supervise_action_field = supervise_action_field
        self.supervise_json_fields = supervise_json_fields
        self.allow_json_integer_null = allow_json_integer_null
        self._decoded_frame_cache: OrderedDict[tuple, tuple[Image.Image, ...]] = OrderedDict()
        self._decoded_frame_cache_size = 64

    def _supervise_turn(self, turn: dict) -> bool:
        if not turn.get("supervise"):
            return False
        return (
            self.supervise_action_stage is None
            or turn.get("action_stage") == self.supervise_action_stage
        )

    def _mark_action_tokens(
        self, input_ids: torch.Tensor, route_mask: torch.Tensor,
        payload: str, start: int, end: int,
    ) -> int:
        """Mark only the generated action value, bounded to its Assistant JSON."""
        try:
            action = str(json.loads(payload).get("action") or "")
        except Exception:
            return 0
        if action not in {"ANSWER", "JOINT_SEMANTIC_SCAN", "TEMPORAL_EXPAND"}:
            return 0
        needle = self.processor.tokenizer(action, add_special_tokens=False)["input_ids"]
        haystack = input_ids[0].tolist()
        for index in range(start, max(start, end - len(needle) + 1)):
            if haystack[index:index + len(needle)] == needle:
                route_mask[:, index:index + len(needle)] = True
                return len(needle)
        raise RuntimeError(f"cannot locate route action tokens for {action}")

    def _mark_action_field_tokens(
        self, input_ids: torch.Tensor, route_mask: torch.Tensor,
        payload: str, start: int, end: int,
    ) -> int:
        """Mark the JSON prefix through the action value, including its key.

        A model that historically emitted a legacy assessment object may know
        the desired action semantically while omitting the mandatory ``action``
        field.  Supervising only the value token cannot repair that prefix.
        """
        import re

        try:
            action = str(json.loads(payload).get("action") or "")
        except Exception:
            return 0
        if action not in {"ANSWER", "JOINT_SEMANTIC_SCAN", "TEMPORAL_EXPAND"}:
            return 0
        match = re.match(r'^\s*\{\s*"action"\s*:\s*"' + re.escape(action) + r'"', payload)
        if not match:
            raise RuntimeError(f"action must be the first JSON field for prefix supervision: {payload[:80]!r}")
        # Locate the value using tokens from the complete payload (there can be
        # a tokenizer merge at the following comma), then supervise every
        # exact full-sequence token from the JSON opening brace through that
        # value.  ``start`` was already located from the complete payload.
        value_mask = torch.zeros_like(route_mask, dtype=torch.bool)
        self._mark_action_tokens(input_ids, value_mask, payload, start, end)
        positions = torch.nonzero(value_mask[0], as_tuple=False).flatten()
        if positions.numel() == 0:
            raise RuntimeError(f"cannot locate route action field tokens for {action}")
        last = int(positions[-1]) + 1
        route_mask[:, start:last] = True
        return last - start

    def __call__(self, rows: list[dict]) -> dict:
        if len(rows) != 1:
            raise ValueError("multimodal SFT currently requires per-device batch size 1")
        row = rows[0]
        if row.get("turns"):
            return self._video_trajectory(row)
        return self._single_state(row)

    def _decode(self, video_path: str, timestamps: list[float], sample_id: str) -> list[Image.Image]:
        path = Path(video_path)
        stat = path.stat()
        key = (
            str(path.resolve()), stat.st_size, stat.st_mtime_ns,
            tuple(float(value) for value in timestamps), self.max_pixels,
        )
        cached = self._decoded_frame_cache.get(key)
        if cached is not None:
            self._decoded_frame_cache.move_to_end(key)
            return list(cached)
        # Teacher review states may intentionally contain the same physical
        # frame once as an initial observation and once as a retrieved hit.
        # Reproduce that visual input instead of silently dropping it.
        frames = recorded_frames(video_path, [float(value) for value in timestamps])
        if len(frames) != len(timestamps):
            # Some old/corrupt containers have inconsistent random-seek
            # indices.  Rebuild the turn one timestamp at a time with an
            # independent FFmpeg fallback and, only when the exact frame is
            # unavailable, use a nearby earlier frame.  This preserves image
            # count/order without borrowing any future tool observation.
            recovered = []
            for raw_timestamp in timestamps:
                timestamp = float(raw_timestamp)
                decoded = recover_frame_near_timestamp(video_path, timestamp)
                if decoded is None:
                    raise RuntimeError(
                        f"{sample_id}: cannot recover frame near timestamp {timestamp:.6f}; "
                        f"batch decoded {len(frames)}/{len(timestamps)} frames"
                    )
                recovered.append(decoded)
            frames = recovered
        images = tuple(resize_to_budget(frame.image, self.max_pixels) for frame in frames)
        self._decoded_frame_cache[key] = images
        self._decoded_frame_cache.move_to_end(key)
        while len(self._decoded_frame_cache) > self._decoded_frame_cache_size:
            self._decoded_frame_cache.popitem(last=False)
        return list(images)

    def _audited_images(self, row: dict) -> list[Image.Image]:
        """Use the exact frozen pixels that the audit reviewed."""
        manifest = row["audit_images"]
        if len(manifest) != len(row["timestamps"]):
            raise RuntimeError(f"{row['id']}: audited image and timestamp counts differ")
        images = []
        for item, timestamp in zip(manifest, row["timestamps"]):
            if abs(float(item["timestamp"]) - float(timestamp)) > 1e-6:
                raise RuntimeError(f"{row['id']}: audited image timestamp changed")
            path = Path(item["path"])
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"{row['id']}: audited image bytes changed: {item['id']}")
            with Image.open(path) as source:
                if source.width * source.height > self.max_pixels:
                    raise RuntimeError(
                        f"{row['id']}: --max-pixels={self.max_pixels} would shrink audited image "
                        f"{item['id']} ({source.width}x{source.height})"
                    )
                images.append(source.convert("RGB"))
        return images

    def _single_state(self, row: dict) -> dict:
        if row.get('contract_version') == 'unified-student-next-step-v1' and (
                self.supervise_action_stage or self.supervise_action_field or self.supervise_json_fields):
            raise ValueError('Unified next-step rows require full supported output supervision')
        if row.get('stage') in {'final_category','policy_payload'} and self.supervise_json_fields:
            validate_classification_context(row['target'],self.supervise_json_fields)
        if self.supervise_action_stage and row.get("stage") != self.supervise_action_stage:
            raise RuntimeError(
                f"{row['id']}: stage {row.get('stage')!r} does not match "
                f"--supervise-action-stage={self.supervise_action_stage!r}"
            )
        images = (self._audited_images(row) if row.get("audit_images") is not None
                  else self._decode(row["video_path"], row["timestamps"], row["id"]))
        content = render_state_content(row, images)
        user_messages = [{"role": "user", "content": content}]
        full_messages = user_messages + [{"role": "assistant", "content": row["target"]}]
        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        if not full_text.startswith(prompt_text):
            raise RuntimeError("chat template assistant prefix mismatch")
        full = self.processor(
            text=[full_text], images=images, padding=False, return_tensors="pt",
        )
        prefix = self.processor(
            text=[prompt_text], images=images, padding=False, return_tensors="pt",
        )
        prefix_length = int(prefix["input_ids"].shape[1])
        if ('request_content_order' in row
                and prefix['input_ids'][0].tolist() != row['request_prompt_token_ids']):
            raise RuntimeError('SFT multimodal prefix differs from actual native request tokens')
        sequence_length = int(full["input_ids"].shape[1])
        if sequence_length > self.max_sequence_length:
            raise RuntimeError(
                f"{row['id']}:{row['stage']} has {sequence_length} tokens, "
                f"above max {self.max_sequence_length}; refusing unsafe multimodal truncation"
            )
        if not torch.equal(full["input_ids"][0, :prefix_length], prefix["input_ids"][0]):
            raise RuntimeError("tokenized prompt is not a prefix of the supervised sample")
        labels = full["input_ids"].clone()
        labels[:, :prefix_length] = -100
        if self.supervise_json_fields:
            payload_ids, selected = token_positions(
                self.processor.tokenizer, row['target'], self.supervise_json_fields,
                self.allow_json_integer_null)
            ids = full['input_ids'][0].tolist()
            matches = [i for i in range(prefix_length, len(ids) - len(payload_ids) + 1)
                       if ids[i:i + len(payload_ids)] == payload_ids]
            if len(matches) != 1:
                raise RuntimeError(f"{row['id']}: JSON target does not have a unique token span")
            labels.fill_(-100)
            for offset in selected:
                labels[0, matches[0] + offset] = full['input_ids'][0, matches[0] + offset]
        if not labels.ne(-100).any():
            raise RuntimeError("assistant target contains no supervised tokens")
        full["labels"] = labels
        route_mask = torch.zeros_like(labels, dtype=torch.bool)
        # A single-state record is rendered with exactly the same final
        # assistant template used by online generation.  Honour
        # ``--route-field-only`` here as well as for trajectory records: the
        # JSON opening/action key are part of the contract, not free-form
        # boilerplate.  Previously this option was silently ignored for
        # single-state data, making a template-aligned router train a weaker
        # objective than the command line stated.
        marker = (self._mark_action_field_tokens
                  if self.supervise_action_field else self._mark_action_tokens)
        route_tokens = marker(
            full["input_ids"], route_mask, row["target"], prefix_length, sequence_length,
        )
        full["route_mask"] = route_mask
        full["sample_id"] = row["id"]
        full["stage"] = row["stage"]
        full["sequence_length"] = sequence_length
        full["target_tokens"] = int(labels.ne(-100).sum())
        full["route_tokens"] = route_tokens
        return full

    def _video_trajectory(self, row: dict) -> dict:
        messages = []
        images_by_message: list[list[Image.Image]] = []
        assistant_turns = []
        supervised_stages = []
        for turn in row["turns"]:
            role = turn["role"]
            if role == "user":
                turn_images = (self._audited_images({"id": row["id"],
                               "timestamps": turn.get("timestamps") or [],
                               "audit_images": turn["audit_images"]})
                               if turn.get("audit_images") is not None else self._decode(
                                   row["video_path"], turn.get("timestamps") or [], row["id"]))
                content = [{"type": "text", "text": turn["text"]}]
                content.extend({"type": "image", "image": image} for image in turn_images)
                messages.append({"role": "user", "content": content})
                images_by_message.append(turn_images)
            elif role == "assistant":
                messages.append({"role": "assistant", "content": turn["content"]})
                images_by_message.append([])
                assistant_turns.append(turn)
                if self._supervise_turn(turn):
                    supervised_stages.append(turn.get("action_stage") or "assistant_action")
            else:
                raise RuntimeError(f"{row['id']}: unsupported trajectory role {role!r}")
        if not supervised_stages:
            raise RuntimeError(f"{row['id']}: trajectory has no supervised assistant action")

        all_images = [image for values in images_by_message for image in values]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        full = self.processor(
            text=[full_text], images=all_images, padding=False, return_tensors="pt",
        )
        sequence_length = int(full["input_ids"].shape[1])
        if sequence_length > self.max_sequence_length:
            raise RuntimeError(
                f"{row['id']}:video_trajectory has {sequence_length} tokens, "
                f"above max {self.max_sequence_length}; refusing unsafe multimodal truncation"
            )
        labels = torch.full_like(full["input_ids"], -100)
        route_mask = torch.zeros_like(labels, dtype=torch.bool)
        temporal_route_mask = torch.zeros_like(labels, dtype=torch.bool)
        route_tokens = 0
        route_marked = False
        action_spans = []
        action_stage_names = []
        full_ids = full["input_ids"][0].tolist()
        search_cursor = 0
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        def find_tokens(needle: list[int], start: int) -> int:
            stop = len(full_ids) - len(needle) + 1
            for index in range(start, max(start, stop)):
                if full_ids[index:index + len(needle)] == needle:
                    return index
            return -1

        # Qwen's template renders historical assistant turns differently from
        # the last assistant turn (thinking prefix rules), so incremental
        # prefix tokenization is not stable. Locate each exact JSON payload in
        # chronological order and supervise only that payload (+ its im_end).
        for turn in assistant_turns:
            content_ids = self.processor.tokenizer(
                turn["content"], add_special_tokens=False,
            )["input_ids"]
            start_index = find_tokens(content_ids, search_cursor)
            if start_index < 0:
                raise RuntimeError(f"{row['id']}: cannot locate assistant JSON tokens")
            end_index = start_index + len(content_ids)
            if end_index < len(full_ids) and full_ids[end_index] == im_end_id:
                end_index += 1
            search_cursor = end_index
            if self._supervise_turn(turn):
                supervision_scope = turn.get("supervision_scope", "full")
                if supervision_scope not in {"full", "action_field"}:
                    raise RuntimeError(
                        f"{row['id']}: unsupported supervision_scope {supervision_scope!r}"
                    )
                action_spans.append([start_index, end_index])
                action_stage = turn.get("action_stage") or "assistant_action"
                action_stage_names.append(action_stage)
                if action_stage == "temporal_route":
                    # GRPO optimizes the post-scan ANSWER-vs-EXPAND choice,
                    # not the category/evidence prose that may share the same
                    # JSON. Keeping this separate from the initial route mask
                    # prevents temporal-policy reward from moving the final
                    # moderation boundary.
                    self._mark_action_tokens(
                        full["input_ids"], temporal_route_mask,
                        turn["content"], start_index, end_index,
                    )
                # Full-trajectory SFT keeps the historical auxiliary route loss
                # on the initial action only.  A targeted repair run can select
                # another action stage (for example the post-scan temporal
                # router), in which case its action token is the route target.
                mark_selected_stage = (
                    self.supervise_action_stage is not None
                    or str(turn.get("action_stage") or "").startswith("initial_")
                )
                if not route_marked and mark_selected_stage:
                    marker = (self._mark_action_field_tokens
                              if self.supervise_action_field or supervision_scope == "action_field"
                              else self._mark_action_tokens)
                    route_tokens = marker(
                        full["input_ids"], route_mask, turn["content"], start_index, end_index,
                    )
                    route_marked = route_tokens > 0
                if supervision_scope == "full":
                    labels[:, start_index:end_index] = full["input_ids"][:, start_index:end_index]
                    if self.supervise_json_fields:
                        payload_ids, selected = token_positions(
                            self.processor.tokenizer, turn['content'], self.supervise_json_fields,
                            self.allow_json_integer_null)
                        if full_ids[start_index:start_index + len(payload_ids)] != payload_ids:
                            raise RuntimeError('field-mask token alignment failed')
                        labels[:, start_index:end_index] = -100
                        for offset in selected:
                            labels[0, start_index + offset] = full['input_ids'][0, start_index + offset]
                else:
                    # Mixed router repair uses the complete TEMPORAL_EXPAND
                    # payload as a positive target, but only the action-field
                    # prefix of ordinary ANSWER negatives.  This regularizes
                    # routing without re-teaching category/severity prose and
                    # prevents a tiny warm-up set from overwriting the parent
                    # model's moderation behavior.
                    if not route_marked:
                        route_tokens = self._mark_action_field_tokens(
                            full["input_ids"], route_mask, turn["content"],
                            start_index, end_index,
                        )
                        route_marked = route_tokens > 0
                    labels[route_mask] = full["input_ids"][route_mask]
        if not labels.ne(-100).any():
            raise RuntimeError(f"{row['id']}: no assistant target tokens after masking")
        full["labels"] = labels
        full["route_mask"] = route_mask
        full["temporal_route_mask"] = temporal_route_mask
        full["sample_id"] = row["id"]
        full["stage"] = "+".join(supervised_stages)
        full["sequence_length"] = sequence_length
        full["target_tokens"] = int(labels.ne(-100).sum())
        full["route_tokens"] = route_tokens
        full["action_spans"] = action_spans
        full["action_stage_names"] = action_stage_names
        return full


def language_lora_targets(model) -> list[str]:
    allowed_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    targets = []
    for name, module in model.named_modules():
        if (name.startswith("model.language_model.layers.")
                and name.rsplit(".", 1)[-1] in allowed_suffixes
                and isinstance(module, torch.nn.Linear)):
            targets.append(name)
    if not targets:
        raise RuntimeError("could not find Qwen3.5 language modules for LoRA")
    return targets


def save_adapter(model, processor, output: Path, state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    (output / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", type=Path, default=None,
        help="Continue training an existing LoRA adapter instead of creating a new one",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--route-loss-weight", type=float, default=0.0,
        help="Auxiliary weight on the mean CE of initial ANSWER/JOINT_SEMANTIC_SCAN action tokens",
    )
    parser.add_argument(
        "--route-only", action="store_true",
        help="Train only initial ANSWER/JOINT_SEMANTIC_SCAN action tokens; used for Router warm-up",
    )
    parser.add_argument(
        "--route-field-only", action="store_true",
        help="With --route-only, supervise the JSON prefix through the action value (including the action key)",
    )
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument('--supervise-json-fields', nargs='+', default=[],
                        help='Supervise only named top-level string fields; preserve complete JSON.')
    parser.add_argument('--allow-json-integer-null',action='store_true',
                        help='With explicit JSON field supervision, also permit integer/null values (evidence indices and unresolved fields).')
    parser.add_argument('--sparse-supervised-logits', action='store_true',
                        help='Compute the LM head only at the exact supervised causal positions.')
    parser.add_argument("--max-sequence-length", type=int, default=12288)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--trajectory-group", choices=("Direct", "Adaptive", "Active"), default=None,
        help="Optionally train only one frozen D/A/A group from a mixed trajectory file",
    )
    parser.add_argument(
        "--supervise-action-stage", default=None,
        help="Mask every Assistant turn except this action_stage (for targeted repair)",
    )
    parser.add_argument("--longest-first", action="store_true",
                        help="Diagnostic option used to stress-test the longest trajectories")
    parser.add_argument("--dataloader-workers", type=int, default=0,
                        help="Keep 0 by default: forking workers after CUDA init can deadlock OpenCV")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optimizer steps; 0 means complete the requested epochs")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.route_only and args.route_loss_weight:
        raise ValueError("--route-only already uses action-token CE; do not combine it with --route-loss-weight")
    if args.route_field_only and not args.route_only:
        raise ValueError("--route-field-only requires --route-only")
    if args.supervise_json_fields and (args.route_only or args.route_loss_weight):
        raise ValueError('JSON field supervision cannot be combined with route loss')
    if args.allow_json_integer_null and not args.supervise_json_fields:
        raise ValueError('--allow-json-integer-null requires --supervise-json-fields')
    # Refuse incompatible objectives BEFORE loading any model or touching CUDA.
    validate_unified_options(args, read_jsonl(args.data))
    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires a CUDA GPU")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = language_lora_targets(model)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=True,
        )
    else:
        config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
        )
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.to("cuda")
    model.train()
    model.print_trainable_parameters()

    dataset = AgentSFTDataset(
        args.data, args.max_examples, args.longest_first, args.trajectory_group,
    )
    if any(r.get('contract_version') == 'unified-student-next-step-v1' for r in dataset.rows):
        if args.route_only or args.supervise_json_fields or args.supervise_action_stage:
            raise ValueError('Unified Student must supervise complete approved actions and parameters')
    generator = torch.Generator().manual_seed(args.seed)
    collator = ResponseOnlyCollator(
        processor, args.max_pixels, args.max_sequence_length,
        args.supervise_action_stage, args.route_field_only,
        tuple(args.supervise_json_fields),
        args.allow_json_integer_null,
    )
    loader_kwargs = {
        "batch_size": 1, "shuffle": True, "generator": generator,
        "num_workers": args.dataloader_workers, "collate_fn": collator,
        "pin_memory": True,
    }
    if args.dataloader_workers:
        loader_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_kwargs)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    requested_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    total_updates = min(requested_updates, args.max_steps) if args.max_steps else requested_updates
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), total_updates,
    )

    run_config = vars(args).copy()
    run_config.update({
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "output": str(args.output.resolve()), "examples": len(dataset),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "updates_per_epoch": updates_per_epoch, "total_optimizer_steps": total_updates,
        "lora_target_modules": len(targets),
        "masked_single_state_suffix_trim": bool(args.route_only or args.supervise_json_fields),
    })
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'json_field_supervision_snapshot.py').write_bytes(Path(__file__).with_name('json_field_supervision.py').read_bytes())
    (args.output / 'trim_sft_suffix_snapshot.py').write_bytes(Path(__file__).with_name('trim_sft_suffix.py').read_bytes())
    (args.output / 'recorded_request_content_snapshot.py').write_bytes(Path(__file__).with_name('recorded_request_content.py').read_bytes())

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    running_loss = 0.0
    running_base_loss = 0.0
    running_route_loss = 0.0
    route_loss_micro_steps = 0
    start = time.time()
    token_max = 0
    untrimmed_token_max = 0
    trim_verified = False
    target_token_max = 0
    epoch = 0
    stop = False
    while not stop and optimizer_step < total_updates:
        epoch += 1
        for batch in loader:
            sample_id = batch.pop("sample_id")
            stage = batch.pop("stage")
            sequence_length = int(batch.pop("sequence_length"))
            target_tokens = int(batch.pop("target_tokens"))
            route_tokens = int(batch.pop("route_tokens"))
            single_state = 'action_spans' not in batch
            batch.pop("action_spans", None)
            batch.pop("action_stage_names", None)
            batch.pop("temporal_route_mask", None)
            untrimmed_token_max = max(untrimmed_token_max, sequence_length)
            route_mask = batch.pop("route_mask")
            if args.route_only:
                batch["labels"] = torch.where(
                    route_mask, batch["labels"], torch.full_like(batch["labels"], -100),
                )
                if not batch["labels"].ne(-100).any():
                    raise RuntimeError(f"{sample_id}: route-only sample has no supervised action token")
            untrimmed_check = None
            if single_state and (args.route_only or args.supervise_json_fields):
                # All images occur in the user prefix of these single states.
                # The later masked explanation cannot affect any causal loss.
                if not trim_verified:
                    untrimmed_check = dict(batch)
                batch, route_mask = trim_masked_suffix(batch, route_mask)
            sequence_length = int(batch['input_ids'].shape[1])
            target_tokens = int(batch['labels'].ne(-100).sum())
            token_max = max(token_max, sequence_length)
            target_token_max = max(target_token_max, target_tokens)
            route_mask = route_mask.to('cuda', non_blocking=True)
            batch = {key: value.to('cuda', non_blocking=True) for key, value in batch.items()}
            if untrimmed_check is not None:
                # Check this exact multimodal model in eval mode before its
                # first optimizer update, in addition to the causal unit test.
                original = {key:value.to('cuda', non_blocking=True) for key,value in untrimmed_check.items()}
                model.eval()
                losses=[]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for probe in [original, batch]:
                        pos, expected_probe = selected_targets(probe['labels'])
                        logits = model(**{k:v for k,v in probe.items() if k!='labels'}, logits_to_keep=pos).logits
                        losses.append(selected_loss(logits, expected_probe).float())
                        del logits
                comparison={'sample_id':sample_id,'untrimmed_tokens':int(original['input_ids'].shape[1]),
                            'trimmed_tokens':sequence_length,'untrimmed_loss':float(losses[0]),
                            'trimmed_loss':float(losses[1]),'absolute_tolerance':.002,'relative_tolerance':.02,
                            'semantic_validation':False}
                try:
                    torch.testing.assert_close(losses[0],losses[1],atol=.002,rtol=.02)
                    comparison['passed']=True
                except AssertionError:
                    comparison['passed']=False
                    raise
                finally:
                    (args.output/'trim_equivalence.json').write_text(json.dumps(comparison,indent=2)+'\n')
                    model.train()
                del original, untrimmed_check, losses
                trim_verified=True
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.sparse_supervised_logits:
                    targets = batch['labels']
                    positions, expected = selected_targets(targets)
                    inputs = {k:v for k,v in batch.items() if k != 'labels'}
                    outputs = model(**inputs, logits_to_keep=positions)
                    base_loss = selected_loss(outputs.logits, expected)
                else:
                    outputs = model(**batch)
                    base_loss = outputs.loss
                loss = base_loss
                route_loss_value = base_loss if args.route_only else None
                if args.route_loss_weight > 0 and route_mask.any():
                    if args.sparse_supervised_logits:
                        # The base CE remains over every approved target token.
                        # This adds weight only to the action value, using the
                        # same sparse causal positions already forwarded.
                        route_selected = route_mask[:, positions + 1]
                        route_loss = selected_subset_loss(outputs.logits, expected, route_selected)
                    else:
                        shifted_logits = outputs.logits[:, :-1, :].contiguous()
                        shifted_labels = batch["labels"][:, 1:].contiguous()
                        shifted_route = route_mask[:, 1:].contiguous()
                        token_losses = F.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.shape[-1]),
                            shifted_labels.view(-1), ignore_index=-100, reduction="none",
                        ).view_as(shifted_labels)
                        route_loss = token_losses[shifted_route].mean()
                    route_loss_value = route_loss
                    loss = loss + args.route_loss_weight * route_loss
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            running_loss += float(loss.detach())
            running_base_loss += float(base_loss.detach())
            if route_loss_value is not None:
                running_route_loss += float(route_loss_value.detach())
                route_loss_micro_steps += 1
            micro_step += 1
            should_update = micro_step % args.gradient_accumulation == 0 or micro_step == len(loader)
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step == 1 or optimizer_step % args.log_steps == 0:
                elapsed = time.time() - start
                mean_loss = running_loss / max(micro_step, 1)
                print(
                    f"step={optimizer_step}/{total_updates} micro={micro_step} "
                    f"loss={mean_loss:.6f} base_loss={running_base_loss/max(micro_step, 1):.6f} "
                    f"route_loss={running_route_loss/max(route_loss_micro_steps, 1):.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"seq={sequence_length} target={target_tokens} id={sample_id} "
                    f"route_tokens={route_tokens} stage={stage} elapsed={elapsed:.1f}s "
                    f"gpu_gb={torch.cuda.max_memory_allocated()/1024**3:.2f}",
                    flush=True,
                )
            state = {
                "optimizer_step": optimizer_step, "micro_step": micro_step,
                "epoch": epoch, "mean_loss": running_loss / max(micro_step, 1),
                "mean_base_loss": running_base_loss / max(micro_step, 1),
                "mean_route_loss": running_route_loss / max(route_loss_micro_steps, 1),
                "route_loss_micro_steps": route_loss_micro_steps,
                "elapsed_seconds": time.time() - start,
                "max_sequence_length_seen": token_max,
                "max_untrimmed_sequence_length_seen": untrimmed_token_max,
                "max_target_tokens_seen": target_token_max,
                "complete": False,
            }
            if args.save_steps and optimizer_step % args.save_steps == 0:
                save_adapter(model, processor, args.output / f"checkpoint-{optimizer_step}", state)
            if optimizer_step >= total_updates:
                stop = True
                break

    final_state = {
        "optimizer_step": optimizer_step, "micro_step": micro_step,
        "epoch": epoch, "mean_loss": running_loss / max(micro_step, 1),
        "mean_base_loss": running_base_loss / max(micro_step, 1),
        "mean_route_loss": running_route_loss / max(route_loss_micro_steps, 1),
        "route_loss_micro_steps": route_loss_micro_steps,
        "elapsed_seconds": time.time() - start,
        "max_sequence_length_seen": token_max,
        "max_untrimmed_sequence_length_seen": untrimmed_token_max,
        "max_target_tokens_seen": target_token_max,
        "complete": optimizer_step == total_updates,
    }
    save_adapter(model, processor, args.output / "final_adapter", final_state)
    (args.output / "COMPLETE").write_text(json.dumps(final_state, ensure_ascii=False) + "\n")
    print("TRAINING_COMPLETE " + json.dumps(final_state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
