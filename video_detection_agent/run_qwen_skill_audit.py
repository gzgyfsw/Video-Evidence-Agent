#!/usr/bin/env python3
"""Run the frozen video-evidence-audit skill with a local Qwen vision server.

prepare is offline. run shares image requests only across same-prefix claims.
The original skill owns prefix construction, acceptance and supervision rules.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_SKILL = Path("/home/user/.codex/skills/video-evidence-audit")
DEFAULT_MODEL_ROOT = HERE.parent.parent / "Qwen3.8-27B"
DEFAULT_MODEL = "evidence-audit-qwen38-27b"
CONFIG = "qwen_runner.json"
SNAPSHOT = "skill_snapshot"
CALLS = "qwen_calls"
REQUIRED = ("SKILL.md", "scripts/audit.py", "references/protocol.md",
            "references/input-contract.md", "references/prompts.md",
            "references/review.schema.json", "references/usage.md",
            "scripts/claim_units.py", "scripts/refine.py", "references/refinement.md")
QUIET_LOCK = threading.RLock()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(skill):
    spec = importlib.util.spec_from_file_location("frozen_video_audit", skill / "scripts/audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quiet(function, args):
    with QUIET_LOCK:
        with contextlib.redirect_stdout(io.StringIO()):
            return function(args)


def endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("--base-url must be an explicit local http://HOST:PORT/v1 endpoint")
    if not parsed.port:
        raise ValueError("--base-url requires an explicit port")
    return value.rstrip("/")


def positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be positive and finite")


def configuration(args):
    for label in ("max_images", "max_tokens"):
        positive(getattr(args, label), label)
    if not 1 <= getattr(args, "claim_batch_size", 1) <= 12:
        raise ValueError("claim_batch_size must be between 1 and 12")
    if getattr(args, "claim_images", "all") not in {"all", "cited"}:
        raise ValueError("claim_images must be all or cited")
    root = args.model_root.resolve()
    if not (root / "config.json").is_file():
        raise ValueError("model root must contain the intended checkpoint's config.json")
    model_files = ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
    return {
        "runner_version": "qwen-skill-audit-v5", "runner_sha256": file_hash(HERE),
        "model": args.model, "backend": "qwen", "base_url": endpoint(args.base_url),
        "expected_model_root": str(root),
        "checkpoint_metadata_hashes": {name: file_hash(root / name) for name in model_files
                                       if (root / name).is_file()},
        "max_images": args.max_images, "max_tokens": args.max_tokens,
        "enable_thinking": args.thinking, "max_attempts_per_task": 2,
        "claim_batch_size": getattr(args, "claim_batch_size", 1),
        "claim_images": getattr(args, "claim_images", "all"),
        "claim_observations": getattr(args, "claim_observations", "compact"),
        "failure_policy": "STOP_ON_FIRST_TECHNICAL_FAILURE; explicit resume only",
        "model_selection": {"basis": "USER_SELECTED_QWEN3.8",
                            "cheapest_or_fastest_verified": False},
        "weak_reference_only": True, "independent_gold": False,
    }


def prepare(args):
    """Freeze the exact skill and use its unmodified prepare implementation."""
    source_skill, out = args.skill_dir.resolve(), args.out.resolve()
    for name in REQUIRED:
        if not (source_skill / name).is_file():
            raise ValueError("missing skill source: " + name)
    core = load_core(source_skill)
    settings = configuration(args)
    claim_plan = getattr(args, "claim_plan", None)
    settings["claim_plan_sha256"] = file_hash(claim_plan) if claim_plan else None
    files = [p for p in sorted(source_skill.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"
             and not any(part.startswith(".") for part in p.relative_to(source_skill).parts)]
    frozen = {str(p.relative_to(source_skill)): p.read_bytes() for p in files}
    settings["skill_source"] = str(source_skill)
    settings["skill_hashes"] = {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
    out.mkdir(parents=True, exist_ok=True)
    with core.lock(out):
        if (out / CONFIG).exists():
            if core.load(out / CONFIG) != settings:
                raise ValueError("skill/model/runner configuration changed; choose a new output directory")
        elif any(out.iterdir()):
            # The lock is the only allowed artifact in a new directory.
            if any(p.name != ".writer.lock" for p in out.iterdir()):
                raise ValueError("output already contains artifacts; do not mix with Codex or another audit")
        for name, data in frozen.items():
            dest = out / SNAPSHOT / name
            if dest.exists() and dest.read_bytes() != data:
                raise ValueError("existing skill snapshot changed")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        runner_snapshot = out / "runner_snapshot.py"
        runner_bytes = HERE.read_bytes()
        if hashlib.sha256(runner_bytes).hexdigest() != settings["runner_sha256"]:
            raise ValueError("runner changed during preparation")
        if runner_snapshot.exists() and runner_snapshot.read_bytes() != runner_bytes:
            raise ValueError("existing runner snapshot changed")
        runner_snapshot.write_bytes(runner_bytes)
        if any((source_skill / name).read_bytes() != data for name, data in frozen.items()):
            raise ValueError("skill changed during snapshot; prepare a new run")
        core.save(out / CONFIG, settings)
    core = load_core(out / SNAPSHOT)
    quiet(core.prepare, argparse.Namespace(input=args.input, out=out, media_root=args.media_root,
          policy=args.policy, policy_version=args.policy_version, replay=args.replay, claim_plan=claim_plan))
    plan = task_plan(out, core)
    preparation = core.load(out / "claim_preparation.json")
    print(json.dumps({"status": "PREPARED_OFFLINE" if preparation["status"] == "READY" else "CLAIM_PREPARATION_REQUIRED",
                      "out": str(out), "tasks": len(plan),
                      "model_calls": 0, "model": settings["model"]}, ensure_ascii=False))


def frozen_run(out):
    config = json.loads((out / CONFIG).read_text(encoding="utf-8"))
    if config["runner_sha256"] != file_hash(HERE):
        raise ValueError("runner changed; use a new versioned output directory")
    if file_hash(out / "runner_snapshot.py") != config["runner_sha256"]:
        raise ValueError("frozen runner changed")
    for name, expected in config["skill_hashes"].items():
        if file_hash(out / SNAPSHOT / name) != expected:
            raise ValueError("frozen skill changed: " + name)
    for name, expected in config["checkpoint_metadata_hashes"].items():
        if file_hash(Path(config["expected_model_root"]) / name) != expected:
            raise ValueError("checkpoint metadata changed: " + name)
    endpoint(config["base_url"])
    core = load_core(out / SNAPSHOT)
    manifest = core.load(out / "manifest.json")
    if manifest["config"]["versions"] != core.version_hashes():
        raise ValueError("prepared run and frozen skill do not match")
    return core, config


def task_plan(out, core):
    plan = []
    order = {"query": 0, "selection_expand": 0, "action": 0, "blind": 1,
             "claim": 2, "stop": 3, "retrieval_effect": 0, "expansion_effect": 0}
    for entry in core.load(out / "manifest.json")["cases"]:
        case = core.verify_case(out, entry)
        local = [(path.parent, core.verify_task(path.parent))
                 for path in sorted((case / "tasks").glob("*/task.json"))]
        plan.extend(sorted(local, key=lambda item: (item[1]["step_index"],
                          order[item[1]["kind"]], item[1]["task_id"])))
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class QwenClient:
    """One HTTP attempt, no SDK retries, conversation state or proxy routing."""
    def __init__(self, config, timeout):
        self.config, self.timeout = config, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, route, body=None):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("QWEN_AUDIT_API_KEY")
        if token:
            headers["Authorization"] = "Bearer " + token
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        req = urllib.request.Request(self.config["base_url"] + route, data=data, headers=headers)
        with self.opener.open(req, timeout=self.timeout) as response:
            return response.read()

    def preflight(self):
        raw = self.request("/models")
        models = json.loads(raw)["data"]
        card = next((m for m in models if m["id"] == self.config["model"]), None)
        if card is None:
            raise ValueError("server does not expose requested Qwen3.8 model; no fallback is allowed")
        if not card.get("root") or Path(card["root"]).resolve() != Path(self.config["expected_model_root"]):
            raise ValueError("server model root does not match the frozen Qwen3.8 checkpoint")
        return raw

    def complete(self, body):
        return self.request("/chat/completions", body)


def extraction(raw, config):
    envelope = json.loads(raw)
    if envelope.get("model") != config["model"]:
        raise ValueError("response came from an unexpected model")
    if len(envelope.get("choices", [])) != 1:
        raise ValueError("expected exactly one reviewer response")
    choice = envelope["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete model response: " + str(choice.get("finish_reason")))
    message = choice["message"]
    if message.get("tool_calls") or message.get("function_call"):
        raise ValueError("reviewer tried to use tools")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("model did not return a text JSON review")
    review = json.loads(content)
    reviews = review.get("reviews", []) if "reviews" in review else [review]
    if not reviews or any(r.get("model") != config["model"] or r.get("backend") != "qwen" for r in reviews):
        raise ValueError("review model/backend does not match the actual request")
    return content.encode(), review, envelope.get("usage")


def events(out, core):
    if getattr(core, "_qwen_ledger", None) is not None:
        return core._qwen_ledger.snapshot()
    return [(p, core.load(p)) for p in sorted((out / CALLS).glob("*/*/*/event.json"))]


def calls_used(out, core):
    if getattr(core, "_qwen_ledger", None) is not None:
        return core._qwen_ledger.used()
    return sum(bool(e.get("dispatched_utc")) for _, e in events(out, core))


class EventLedger:
    """Index actual dispatch events once per locked run, not once per task.

    Disk artifacts are still authoritative at startup. Cache validation keeps
    hashing actual request/response bytes; only ledger lookups are accelerated.
    """
    def __init__(self, records):
        self.lock = threading.RLock()
        self.records, self.by_task, self.count = {}, {}, 0
        for path, event in records:
            self.update(path, event)

    def update(self, path, event):
        with self.lock:
            old = self.records.get(path)
            self.count += int(bool(event.get("dispatched_utc"))) - int(bool(old and old.get("dispatched_utc")))
            self.records[path] = copy.deepcopy(event)
            for tid in event_tasks(event):
                self.by_task.setdefault((event["case_id"], tid), set()).add(path)

    def used(self):
        with self.lock:
            return self.count

    def snapshot(self, key=None):
        with self.lock:
            paths = self.records if key is None else self.by_task.get(key, ())
            return [(p, copy.deepcopy(self.records[p])) for p in sorted(paths)]


def save_event(core, path, event):
    core.save(path, event)
    if getattr(core, "_qwen_ledger", None) is not None:
        core._qwen_ledger.update(path, event)


def task_events(out, core, case_id, task_id):
    if getattr(core, "_qwen_ledger", None) is not None:
        return core._qwen_ledger.snapshot((case_id, task_id))
    return [(p, e) for p, e in events(out, core) if e["case_id"] == case_id and task_id in event_tasks(e)]


def event_tasks(event):
    return event.get("task_ids", [event["task_id"]])


def review_artifact(event, task_id):
    if event.get("task_ids"):
        return task_id + ".response.json", event.get("review_raw_sha256_by_task", {}).get(task_id)
    return "response.json", event.get("review_raw_sha256")


def selected_images(task, config):
    images = task["input"]["images"]
    if task["kind"] != "claim" or config.get("claim_images", "all") != "cited":
        return images
    ids = set(task["input"]["subject"].get("frame_ids", []))
    # Missing/unknown citations remain for the skill to constrain; never drop
    # legitimate citations to fit an image or context limit.
    if not ids:
        return images
    if not ids <= {im["id"] for im in images}:
        raise ValueError("claim cites unavailable images")
    return [im for im in images if im["id"] in ids]


def shared_prefix(task):
    return {k: v for k, v in task["input"].items() if k not in {"task_id", "subject"}}


def request_groups(plan, config):
    """Batch only claims with identical original prefixes; keep time order."""
    size = config.get("claim_batch_size", 1)
    groups, current = [], []
    for item in plan:
        task = item[1]
        if current and (len(current) >= size or task["kind"] != "claim"
                        or current[0][1]["kind"] != "claim"
                        or shared_prefix(task) != shared_prefix(current[0][1])
                        or task["prerequisites"] != current[0][1]["prerequisites"]):
            groups.append(current)
            current = []
        current.append(item)
        if task["kind"] != "claim":
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def group_images(tasks, config):
    wanted = {im["id"] for t in tasks for im in selected_images(t, config)}
    images = [im for im in tasks[0]["input"]["images"] if im["id"] in wanted]
    if len(images) > config["max_images"]:
        raise ValueError("too many images; attachments are never truncated")
    return images


def group_request(tasks, core, config):
    if any(t["kind"] != "claim" or shared_prefix(t) != shared_prefix(tasks[0]) for t in tasks):
        raise ValueError("cannot batch different kinds, prompts or causal prefixes")
    images = group_images(tasks, config)
    parts = core.prompts()
    prompt = (parts["common"] + "\n\n" + parts["claim"]
              + "\n共享前缀输入：" + json.dumps(shared_prefix(tasks[0]), ensure_ascii=False)
              + "\n逐项审核任务：" + json.dumps([
                  {"task_id": t["task_id"], "input_sha256": t["input_sha256"],
                   "subject": t["input"]["subject"], "constraints": t["constraints"]} for t in tasks], ensure_ascii=False)
              + "\n每项原input等于共享前缀加其task_id和subject。仅共享图片请求，不合并判断。"
              "返回reviews数组，每个task_id恰好一项；保持其原input_sha256，独立判断和引用。"
              "不得把其他主张或Teacher文字当作图片证据。observations只记录该项所需的真实图片，避免重复全前缀描述。"
              "说明用简短中文，聚焦该项证据和局限；不重复政策全文和其他主张的描述。"
              + "\n实际附件ID：" + json.dumps([im["id"] for im in images])
              + "\n审核运行元数据：" + json.dumps({"model": config["model"], "backend": "qwen"})
              + contract_reminder(tasks[0]))
    content = [{"type": "text", "text": prompt}]
    import base64
    from PIL import Image
    for im in images:
        if file_hash(im["path"]) != im["sha256"]:
            raise ValueError("image changed")
        with Image.open(im["path"]) as picture:
            fmt = picture.format.lower()
        mime = "image/jpeg" if fmt == "jpeg" else "image/" + fmt
        content.extend([{"type": "text", "text": "image " + im["id"]},
                        {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64,"
                         + base64.b64encode(Path(im["path"]).read_bytes()).decode()}}])
    # Each object's task/hash pair is bound by local validation below.
    items = [wire_schema(t, core, config)["json_schema"]["schema"] for t in tasks]
    for schema in items:
        for key in ("images_viewed", "evidence_frame_ids"):
            schema["properties"][key]["items"] = {"enum": [im["id"] for im in images]}
            # vLLM's structured-output adapter drops JSON Schema's
            # ``uniqueItems`` keyword.  Bounding these set-like arrays to the
            # number of real attached images prevents the common failure mode
            # where the model repeats frame IDs and the local validator then
            # rejects an otherwise usable response.  The raw response remains
            # immutable and is still validated locally after generation.
            schema["properties"][key]["maxItems"] = len(schema["properties"][key].get("items", {}).get("enum", []))
        if images:
            schema["properties"]["observations"]["items"]["properties"]["frame_id"] = {"enum": [im["id"] for im in images]}
            schema["properties"]["observations"]["maxItems"] = len(images)
    # vLLM supports ordinary array items; exact identity/hash pairing and
    # completeness are still checked locally before any accept is published.
    alternatives = []
    for task, schema in zip(tasks, items):
        non_support = copy.deepcopy(schema)
        non_support["properties"]["judgment"] = {"enum": ["UNDETERMINED"] if task["constraints"]
                                                 else ["NOT_SUPPORTED", "UNDETERMINED"]}
        alternatives.append(non_support)
        if not task["constraints"]:
            support = copy.deepcopy(schema)
            support["properties"]["judgment"] = {"enum": ["SUPPORTED"]}
            cited = task["input"]["subject"].get("frame_ids", [])
            if cited:
                support["properties"]["images_viewed"] = {"type": "array", "minItems": len(cited),
                                                           "maxItems": len(cited), "items": {"enum": cited}}
                if task["input"]["subject"].get("kind") in {"fact", "interpretation"}:
                    support["properties"]["evidence_frame_ids"] = {"type": "array", "minItems": len(cited),
                                                                   "maxItems": len(cited), "items": {"enum": cited}}
            alternatives.append(support)
    # Every union branch is a COMPLETE review schema. Some vLLM grammar
    # backends treat sparse anyOf refinements as the whole object and otherwise
    # drop required identity/judgment fields despite outer schema requirements.
    item = {"anyOf": alternatives}
    schema = {"type": "object", "additionalProperties": False, "required": ["reviews"],
              "properties": {"reviews": {"type": "array", "minItems": len(tasks), "maxItems": len(tasks), "items": item}}}
    if config.get("claim_observations", "compact") == "compact":
        content[0]["text"] += "\nclaim的observations填空数组[]；请在reason中简述该项可见事实和局限，证据帧列于evidence_frame_ids。完整逐图观察已由独立blind单独留档。"
    return {"model": config["model"], "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": min(16384, config["max_tokens"] * len(tasks)),
            "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "video_evidence_reviews", "strict": True, "schema": schema}}}, images


def wire_schema(task, core, config):
    """Guide output shape without replacing the full original local validator.

    The transport subset is the same kind of projection used by the Codex CLI
    runner: conditional/uniqueness checks still happen in skill.accept.
    """
    def project(value):
        if isinstance(value, list):
            return [project(v) for v in value]
        if not isinstance(value, dict):
            return value
        return {k: project(v) for k, v in value.items()
                if k not in {"$schema", "allOf", "uniqueItems", "pattern", "minLength"}}
    schema = project(core.load(core.REF / "review.schema.json"))
    props = schema["properties"]
    if task["kind"] == "claim" and config.get("claim_observations", "compact") == "compact":
        props["observations"]["maxItems"] = 0
    for name, value in (("task_id", task["task_id"]), ("input_sha256", task["input_sha256"]),
                        ("model", config["model"]), ("backend", "qwen")):
        props[name] = {"enum": [value]}
    ids = [image["id"] for image in task["input"]["images"]]
    if ids:
        for name in ("evidence_frame_ids", "images_viewed"):
            props[name]["items"] = {"enum": ids}
            # See the batching path above: the server currently ignores
            # uniqueItems, so a hard upper bound stops repeated frame IDs from
            # consuming an attempt before local schema validation.
            props[name]["maxItems"] = len(ids)
        props["observations"]["items"]["properties"]["frame_id"] = {"enum": ids}
        props["observations"]["maxItems"] = len(ids)
    else:
        for name in ("evidence_frame_ids", "images_viewed", "observations"):
            props[name]["maxItems"] = 0
    tool_ids = allowed_tools(task)
    if tool_ids:
        props["tool_step_ids"]["items"] = {"enum": tool_ids}
    else:
        props["tool_step_ids"]["maxItems"] = 0
    if task["constraints"]:
        props["judgment"] = {"enum": ["UNDETERMINED"]}
    return {"type": "json_schema", "json_schema": {
        "name": "video_evidence_review", "strict": True, "schema": schema}}


def allowed_tools(task):
    ids, calls = set(), set()
    for event in task["input"]["prefix_events"]:
        if event["type"] == "decision" and event.get("action") != "ANSWER":
            calls.add(event["step_id"])
        elif event["type"] == "tool_result":
            ids.add(event["step_id"])
            if event.get("call_step_id") in calls:
                ids.add(event["call_step_id"])
    return sorted(ids)


def contract_reminder(task):
    """Describe existing validator rules, never new semantic/category policy."""
    ids = [image["id"] for image in task["input"]["images"]]
    rules = {
        "来源": "原版 skill 的 validate_review 和 prerequisite_completed，非新增分类政策",
        "真实附件ID": ids,
        "实际查看": "images_viewed仅记录本次实际查看的附件ID；claim支持要求打开全部原引用帧，局部主张无需重复打开未引用历史帧。expansion_effect支持要求打开全部实际返回帧；blind/动作/stop支持须查看全部供应前缀图。",
        "可引用工具ID": allowed_tools(task),
        "工具规则": "tool_step_ids 仅可使用上列ID；initial、observe和当前待审动作不是可引用的历史调用。没有可引用工具时填空数组。",
        "观察格式": "observations 每项为 frame_id、visible_fact、uncertain；uncertain必须是布尔值，说明文字放visible_fact或reason；evidence_frame_ids、images_viewed、tool_step_ids 以及 observations 的 frame_id 均不得重复。",
        "盲观察": "blind必须对每张附件各记录一条观察；允许UNDETERMINED，不要求SUPPORTED。",
        "主张引用": "fact/interpretation若SUPPORTED，evidence_frame_ids覆盖全部原引用帧；category/severity/disposition的证据可为原引用帧子集，但实际查看仍覆盖全部原引用帧。不能用邻帧替代。",
        "判断": "不支持或无法判断是合法完成结果；不得为了通过格式验证改变语义判断。",
    }
    return "\n接口验收规则：" + json.dumps(rules, ensure_ascii=False)


def cached(folder, task, core, config, out):
    if not (folder / "review.json").exists():
        return False
    saved = core.load(folder / "review.json")
    core.effective(task, folder)
    answer = saved["response"]
    if answer["model"] != config["model"] or answer["backend"] != "qwen":
        raise ValueError("cannot mix another model's cached reviews into this run")
    if answer["status"] == "TECHNICAL_FAILURE":
        return False
    case_id = folder.parents[1].name
    matching = [(p, e) for p, e in task_events(out, core, case_id, task["task_id"])
                if e.get("status") in {"VALIDATED", "TECHNICAL_FAILURE"}]
    matching = [(p, e) for p, e in matching if review_artifact(e, task["task_id"])[1] == saved["raw_sha256"]
                and e.get("request_sha256") == saved.get("actual_request_sha256")]
    if not matching:
        raise ValueError("accepted review is missing its matching Qwen request/event provenance")
    path, event = matching[-1]
    for name, expected in (("request.json", event["request_sha256"]),
                           (review_artifact(event, task["task_id"])[0], saved["raw_sha256"]),
                           ("model-response.raw", saved["model_raw_sha256"])):
        if file_hash(path.parent / name) != expected:
            raise ValueError("cached actual request/response artifact changed: " + name)
    return True


def close_saved_call(call, folder, task, core, config, event):
    """Recover a fully saved reply without ever reissuing the request."""
    if file_hash(call / "request.json") != event["request_sha256"]:
        raise ValueError("saved actual request changed")
    raw = (call / "model-response.raw").read_bytes()
    review_path = call / "response.json"
    extracted = False
    try:
        content, review, usage = extraction(raw, config)
        extracted = True
        review_path.write_bytes(content)  # Exact content; no JSON repair or judgment changes.
        if not set(review.get("images_viewed", [])) <= {im["id"] for im in event["image_manifest"]}:
            (call / "extracted-review.json").write_bytes(content)
            review_path.write_bytes(b"ACTUAL_IMAGE_ATTACHMENT_MISMATCH\n" + content)
            raise ValueError("review claims to view an image not actually attached")
        # accept writes its validated attempt before publishing review.json.
        # Restore that identical record if interrupted between the two writes.
        record_path = folder / "attempts" / f"{event['attempt']:02d}.json"
        if not (folder / "review.json").exists() and record_path.exists():
            record = core.load(record_path)
            if (record.get("status") == "VALIDATED" and record.get("raw_sha256") == file_hash(review_path)
                    and record.get("actual_request_sha256") == event["request_sha256"]):
                core.validate_review(task, record["response"])
                if any(not core.prerequisite_completed(folder, req) for req in task["prerequisites"]):
                    raise ValueError("saved acceptance lost its blind prerequisite")
                with core.lock(folder):
                    core.save(folder / "review.json", record)
        quiet(core.accept, argparse.Namespace(task=folder, response=review_path,
              raw_response=call / "model-response.raw", request=call / "request.json"))
        event.update(status="VALIDATED", judgment=review["judgment"],
                     review_status=review["status"], usage=usage,
                     review_raw_sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        # If extraction failed, feed the unchanged raw envelope to accept. It
        # fails its review schema and leaves a genuine technical-failure record.
        if not extracted:
            review_path.write_bytes(b"BACKEND_RESPONSE_FAILURE\n" + raw)
        records = core._attempt_records(folder)
        if len(records) < event["attempt"]:
            try:
                quiet(core.accept, argparse.Namespace(task=folder, response=review_path,
                      raw_response=call / "model-response.raw", request=call / "request.json"))
            except Exception:
                pass
        event.update(status="TECHNICAL_FAILURE", error=type(exc).__name__ + ": " + str(exc))
    event["ended_utc"] = core.utc()
    save_event(core, call / "event.json", event)
    return event


def close_saved_group(call, group, core, config, event):
    """Accept each original object against its schema; retain the raw batch."""
    if file_hash(call / "request.json") != event["request_sha256"]:
        raise ValueError("saved actual request changed")
    raw = (call / "model-response.raw").read_bytes()
    validated, errors = dict(event.get("review_raw_sha256_by_task", {})), []
    try:
        if event.get("transport_error"):
            raise ValueError(event["transport_error"])
        content, answer, usage = extraction(raw, config)
        (call / "response.json").write_bytes(content)
        values = answer.get("reviews")
        if not isinstance(values, list):
            raise ValueError("batch response must contain reviews")
        returned = {r["task_id"]: r for r in values}
        if len(returned) != len(values) or set(returned) != {t["task_id"] for _, t in group}:
            raise ValueError("batch omitted, duplicated or invented a task")
        event["usage"] = usage
    except Exception as exc:
        returned = {}
        errors.append(str(exc))
    attached = {im["id"] for im in event["image_manifest"]}
    for folder, task in group:
        tid = task["task_id"]
        response_path = call / (tid + ".response.json")
        try:
            if tid not in returned:
                raise ValueError("invalid batch envelope: " + "; ".join(errors))
            review = returned[tid]
            core.validate_review(task, review)
            if not set(review["images_viewed"]) <= attached:
                core.save(call / (tid + ".extracted-review.json"), review)
                response_path.write_bytes(b"ACTUAL_IMAGE_ATTACHMENT_MISMATCH\n" + json.dumps(review).encode())
                raise ValueError("review claims to view an image not actually attached")
            # Check prerequisites and all source image hashes in original
            # skill.accept, never infer acceptance from the batch schema.
            if not response_path.exists():
                core.save(response_path, review)
            elif core.load(response_path) != review:
                raise ValueError("saved per-object response changed")
            records = core._attempt_records(folder)
            if not (folder / "review.json").exists() and records:
                record = core.load(records[-1])
                if record.get("status") == "VALIDATED" and record.get("raw_sha256") == file_hash(response_path) \
                        and record.get("actual_request_sha256") == event["request_sha256"]:
                    with core.lock(folder):
                        core.save(folder / "review.json", record)
            quiet(core.accept, argparse.Namespace(task=folder, response=response_path,
                  raw_response=call / "model-response.raw", request=call / "request.json"))
            if review["status"] != "COMPLETED":
                raise ValueError("reviewer reported technical failure")
            validated[tid] = file_hash(response_path)
        except Exception as exc:
            errors.append(tid + ": " + str(exc))
            # Record the actual failed content (not a fabricated verdict),
            # preserving two-attempt accounting even for malformed envelopes.
            if not response_path.exists():
                response_path.write_bytes(b"BACKEND_BATCH_FAILURE\n" + raw)
            if len(core._attempt_records(folder)) < event["attempt_by_task"][tid]:
                try:
                    quiet(core.accept, argparse.Namespace(task=folder, response=response_path,
                          raw_response=call / "model-response.raw", request=call / "request.json"))
                except Exception:
                    pass
    event.update(status="TECHNICAL_FAILURE" if errors else "VALIDATED", errors=errors,
                 review_raw_sha256_by_task=validated, ended_utc=core.utc())
    save_event(core, call / "event.json", event)
    return event


def run_group(out, group, core, config, client, max_calls, dispatch_lock, stop):
    pending = [(f, t) for f, t in group if not cached(f, t, core, config, out)]
    if not pending:
        return [(f, t, "CACHED_VALID") for f, t in group]
    if len(pending) == 1:
        f, t = pending[0]
        status = run_one(out, f, t, core, config, client, max_calls, dispatch_lock, stop)
        return [(folder, task, status if folder == f else "CACHED_VALID") for folder, task in group]
    for folder, task in pending:
        if any(not core.prerequisite_completed(folder, req) for req in task["prerequisites"]):
            raise RuntimeError("blind prerequisite not complete")
        if len(core._attempt_records(folder)) >= 2:
            raise RuntimeError("two model attempts per object exhausted")
    tasks = [t for _, t in pending]
    body, images = group_request(tasks, core, config)
    case_id = pending[0][0].parents[1].name
    batch_id = "batch-" + core.digest([t["input_sha256"] for t in tasks])[:24]
    parent = out / CALLS / case_id / batch_id
    number = len(list(parent.glob("*/event.json"))) + 1
    call = parent / f"{number:02d}"
    call.mkdir(parents=True, exist_ok=True)
    if (call / "request.json").exists() and core.load(call / "request.json") != body:
        raise ValueError("existing unsent batch request changed")
    if (call / "model-response.raw").exists():
        raise ValueError("orphan batch response without reserved event; inspect before dispatch")
    core.save(call / "request.json", body)
    event = {"case_id": case_id, "task_id": batch_id, "task_ids": [t["task_id"] for t in tasks],
             "input_sha256_by_task": {t["task_id"]: t["input_sha256"] for t in tasks},
             "attempt_by_task": {t["task_id"]: len(core._attempt_records(f)) + 1 for f, t in pending},
             "kind": "claim", "attempt": number, "status": "DISPATCHING", "model": config["model"],
             "request_sha256": file_hash(call / "request.json"), "image_manifest": images,
             "context": "fresh shared prefix; independent original object judgments"}
    with dispatch_lock:
        if stop.is_set():
            raise RuntimeError("stopped after technical failure; request not dispatched")
        if calls_used(out, core) >= max_calls:
            raise RuntimeError("global model-call budget exhausted")
        event["dispatched_utc"] = core.utc()
        save_event(core, call / "event.json", event)
    started = time.monotonic()
    try:
        raw = client.complete(body)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        event.update(http_status=exc.code, transport_error="HTTPError: " + str(exc))
    except Exception as exc:
        raw = (type(exc).__name__ + ": " + str(exc)).encode()
        event["transport_error"] = raw.decode()
    (call / "model-response.raw").write_bytes(raw)
    event["elapsed_seconds"] = round(time.monotonic() - started, 3)
    event = close_saved_group(call, pending, core, config, event)
    if event["status"] != "VALIDATED":
        stop.set()
        raise RuntimeError("technical failure in claim batch; inspect artifacts and explicitly resume")
    return [(f, t, "VALIDATED" if (f, t) in pending else "CACHED_VALID") for f, t in group]


def recover_saved_calls(out, core, config):
    for path, event in events(out, core):
        if event["status"] != "DISPATCHING":
            continue
        call = path.parent
        if not (call / "model-response.raw").exists():
            raise RuntimeError("uncertain interrupted request; do not resend automatically: " + str(call))
        if event.get("task_ids"):
            group = []
            for tid in event["task_ids"]:
                folder = out / "cases" / event["case_id"] / "tasks" / tid
                task = core.verify_task(folder)
                if task["input_sha256"] != event["input_sha256_by_task"][tid]:
                    raise ValueError("interrupted batch no longer matches input")
                group.append((folder, task))
            recovered = close_saved_group(call, group, core, config, event)
            if recovered["status"] != "VALIDATED":
                raise RuntimeError("recovered batch has technical failures; explicit resume required")
            continue
        folder = out / "cases" / event["case_id"] / "tasks" / event["task_id"]
        task = core.verify_task(folder)
        if task["input_sha256"] != event["input_sha256"]:
            raise ValueError("interrupted call no longer matches its input")
        close_saved_call(call, folder, task, core, config, event)


def run_one(out, folder, task, core, config, client, max_calls, dispatch_lock, stop=None):
    if cached(folder, task, core, config, out):
        return "CACHED_VALID"
    for required in task["prerequisites"]:
        if not core.prerequisite_completed(folder, required):
            return "SKIPPED_BLIND_PREREQUISITE"
    parent = out / CALLS / folder.parents[1].name / task["task_id"]
    for attempt in (1, 2):
        call = parent / f"{attempt:02d}"
        if (call / "event.json").exists():
            event = core.load(call / "event.json")
            if event["status"] == "DISPATCHING":
                if not (call / "model-response.raw").exists():
                    raise RuntimeError("uncertain interrupted request; do not resend automatically: " + str(call))
                event = close_saved_call(call, folder, task, core, config, event)
            if event["status"] == "VALIDATED" and event["review_status"] == "COMPLETED":
                if not cached(folder, task, core, config, out):
                    raise ValueError("accepted reply was removed; use a new run")
                return "CACHED_VALID"
            if event["status"] in {"VALIDATED", "TECHNICAL_FAILURE"}:
                continue
        if len(core._attempt_records(folder)) >= attempt:
            prior_group = [e for _, e in task_events(out, core, folder.parents[1].name, task["task_id"]) if e.get("task_ids")
                           and e.get("attempt_by_task", {}).get(task["task_id"]) == attempt]
            if prior_group and all(e["status"] == "TECHNICAL_FAILURE" for e in prior_group):
                continue
            raise ValueError("unmatched accept attempt; do not overwrite prior artifacts")
        call.mkdir(parents=True, exist_ok=True)
        try:
            quiet(core.export_qwen, argparse.Namespace(task=folder, model=config["model"],
                  output=call / "skill-export.json", max_images=config["max_images"], max_tokens=config["max_tokens"]))
        except Exception as exc:
            core.save(parent / "blocked.json", {"status": "BLOCKED_BEFORE_REQUEST", "error": str(exc),
                      "task_id": task["task_id"], "images_not_truncated": True, "model_calls": 0})
            raise
        body = core.load(call / "skill-export.json")
        body["chat_template_kwargs"] = {"enable_thinking": config["enable_thinking"]}
        body["response_format"] = wire_schema(task, core, config)
        body["messages"][0]["content"][0]["text"] += contract_reminder(task)
        if task["kind"] == "claim" and config.get("claim_observations", "compact") == "compact":
            body["messages"][0]["content"][0]["text"] += "\nclaim的observations填[]；在reason中简述该项可见事实和局限，证据帧列于evidence_frame_ids；完整逐图观察由blind留档。"
        chosen = selected_images(task, config)
        if chosen != task["input"]["images"]:
            # Keep full original input and its hash; send every cited image
            # while recording the smaller actual attachment manifest.
            chosen_ids = {im["id"] for im in chosen}
            original_content = body["messages"][0]["content"]
            reduced = [original_content[0]]
            for index, im in enumerate(task["input"]["images"]):
                if im["id"] in chosen_ids:
                    reduced.extend(original_content[1 + 2*index:3 + 2*index])
            body["messages"][0]["content"] = reduced
            reduced[0]["text"] += "\n实际附件ID：" + json.dumps(sorted(chosen_ids))
        # The exported semantic prompt and image order are never rewritten.
        if len(body["messages"]) != 1 or body["messages"][0]["role"] != "user":
            raise ValueError("request must have a fresh single-task context")
        if (call / "request.json").exists() and core.load(call / "request.json") != body:
            raise ValueError("existing unsent request changed")
        core.save(call / "request.json", body)
        event = {"case_id": folder.parents[1].name, "task_id": task["task_id"],
                 "kind": task["kind"], "attempt": attempt, "status": "DISPATCHING",
                 "dispatched_utc": core.utc(), "model": config["model"],
                 "input_sha256": task["input_sha256"], "request_sha256": file_hash(call / "request.json"),
                 "image_manifest": chosen, "context": "fresh single user message"}
        with dispatch_lock:
            if stop is not None and stop.is_set():
                raise RuntimeError("stopped after technical failure; request not dispatched")
            if calls_used(out, core) >= max_calls:
                raise RuntimeError("global model-call budget exhausted")
            save_event(core, call / "event.json", event)  # Reserve budget before the network boundary.
        started = time.monotonic()
        try:
            raw = client.complete(body)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            event.update(http_status=exc.code, transport_error="HTTPError: " + str(exc))
        except Exception as exc:
            # A transport failure is not a fabricated model answer.
            raw = (type(exc).__name__ + ": " + str(exc)).encode()
            event["transport_error"] = raw.decode()
        (call / "model-response.raw").write_bytes(raw)
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if "transport_error" in event:
            # Retain even a JSON-shaped HTTP error as failure, never as success.
            (call / "response.json").write_bytes(b"TRANSPORT_FAILURE\n" + raw)
            try:
                quiet(core.accept, argparse.Namespace(task=folder, response=call / "response.json",
                      raw_response=call / "model-response.raw", request=call / "request.json"))
            except Exception:
                pass
            event.update(status="TECHNICAL_FAILURE", error=event["transport_error"], ended_utc=core.utc())
            save_event(core, call / "event.json", event)
        else:
            event = close_saved_call(call, folder, task, core, config, event)
        if event["status"] == "VALIDATED" and event["review_status"] == "COMPLETED":
            return "VALIDATED"
        if stop is not None:
            stop.set()
        raise RuntimeError("technical failure; inspect artifacts and explicitly resume (at most two attempts)")
    return "TECHNICAL_FAILURE"


@contextlib.contextmanager
def runner_lock(out):
    path = out / ".qwen-runner.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink()


def run(args):
    out = args.out.resolve()
    positive(args.max_calls, "max_calls")
    positive(args.timeout, "timeout")
    workers = getattr(args, "workers", 1)
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    core, config = frozen_run(out)
    with runner_lock(out):
        core._qwen_ledger = EventLedger(events(out, core))
        plan = task_plan(out, core)
        if any("CLAIM_PREPARATION_REQUIRED" in task["constraints"] for _, task in plan):
            raise ValueError("CLAIM_PREPARATION_REQUIRED: complete the input-bound claim plan before any model request")
        results, error = [], None
        dispatch_lock, progress_lock, stop = threading.Lock(), threading.Lock(), threading.Event()
        history = out / "executions"
        history.mkdir(exist_ok=True)
        execution = history / f"{len(list(history.glob('[0-9]*')))+1:04d}"
        execution.mkdir()
        by_case, completed_cases, case_counts = {}, set(), Counter()
        for folder, task in plan:
            by_case.setdefault(folder.parents[1].name, []).append((folder, task))
        try:
            recover_saved_calls(out, core, config)
            # Validate all caches before any HTTP request.
            pending = [folder for folder, task in plan if not cached(folder, task, core, config, out)]
            if pending:
                models_raw = QwenClient(config, args.timeout).preflight()
                (execution / "server-models.raw.json").write_bytes(models_raw)
            def run_case(case_plan):
                # A worker owns exactly one case's chronological task sequence.
                # Each model call still contains one independent user message.
                client = QwenClient(config, args.timeout)
                try:
                    for group in request_groups(case_plan, config):
                        if stop.is_set():
                            break
                        group_results = run_group(out, group, core, config, client,
                                                  args.max_calls, dispatch_lock, stop)
                        with progress_lock:
                            for folder, task, status in group_results:
                                result = {"case_id": folder.parents[1].name, "task_id": task["task_id"],
                                          "kind": task["kind"], "status": status}
                                results.append(result)
                                case_counts[result["case_id"]] += 1
                            case_id = case_plan[0][0].parents[1].name
                            if case_counts[case_id] == len(case_plan):
                                completed_cases.add(case_id)
                            progress = {"closed_tasks": len(results), "total_tasks": len(plan),
                                        "model_calls": calls_used(out, core),
                                        "completed_videos": len(completed_cases), "total_videos": len(by_case),
                                        "current_trajectory_id": task["trajectory_id"], **result}
                            core.save(out / "qwen_progress.json", {"updated_utc": core.utc(), **progress})
                            print(json.dumps(progress, ensure_ascii=False), flush=True)
                except Exception:
                    stop.set()
                    raise

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_case, case_plan) for case_plan in by_case.values()]
                errors = []
                try:
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            errors.append(type(exc).__name__ + ": " + str(exc))
                except KeyboardInterrupt:
                    stop.set()  # Drain existing requests; never strand an uncertain call.
                    errors.append("INTERRUPTED_AFTER_DRAINING_ACTIVE_REQUESTS")
                if errors:
                    raise RuntimeError("; ".join(errors))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        finally:
            quiet(core.summarize, argparse.Namespace(out=out))
            failed = any(r["status"] in {"TECHNICAL_FAILURE", "SKIPPED_BLIND_PREREQUISITE"} for r in results)
            report = {"status": "STOPPED" if error else "CLOSED_WITH_TECHNICAL_FAILURES" if failed else "CLOSED", "error": error,
                      "model": config["model"], "total_tasks": len(plan), "closed_tasks": len(results),
                      "model_calls": calls_used(out, core), "max_calls": args.max_calls,
                      "max_attempts_per_task": 2, "timeout_seconds": args.timeout,
                      "workers": workers, "parallelism": "independent cases; sequential within each case",
                      "claim_batch_size": config.get("claim_batch_size", 1),
                      "claim_images": config.get("claim_images", "all"),
                      "claim_observations": config.get("claim_observations", "compact"),
                      "completed_videos": len(completed_cases), "total_videos": len(by_case),
                      "status_counts": dict(Counter(r["status"] for r in results)), "results": results,
                      "weak_reference_only": True, "independent_gold": False, "training_started": False}
            core.save(out / "qwen_execution.json", report)
            core.save(execution / "execution.json", report)
        if error:
            raise RuntimeError(error)
        if any(r["status"] in {"TECHNICAL_FAILURE", "SKIPPED_BLIND_PREREQUISITE"} for r in results):
            raise RuntimeError("run closed with technical failures or blocked blind prerequisites; inspect summary")
        print(json.dumps({"status": "CLOSED", "model_calls": report["model_calls"],
                          "summary": str(out / "summary.json")}, ensure_ascii=False))


def inspect_run(args):
    out = args.out.resolve()
    core, config = frozen_run(out)
    plan = task_plan(out, core)
    groups = request_groups(plan, config)
    print(json.dumps({"model": config["model"], "expected_model_root": config["expected_model_root"],
                      "tasks": len(plan), "model_calls": calls_used(out, core),
                      "task_kinds": dict(Counter(t["kind"] for _, t in plan)),
                      "max_images_in_task": max((len(t["input"]["images"]) for _, t in plan), default=0),
                      "planned_http_calls": len(groups),
                      "claim_batch_size": config.get("claim_batch_size", 1),
                      "original_image_attachments": sum(len(t["input"]["images"]) for _, t in plan),
                      "planned_image_attachments": sum(len(group_images([t for _, t in g], config)) for g in groups),
                      "weak_reference_only": True}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="offline skill snapshot and mechanical preparation")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--media-root", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--policy-version")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--claim-plan", type=Path, help="explicit claim decomposition plan, applied by the shared skill")
    p.add_argument("--skill-dir", type=Path, default=DEFAULT_SKILL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument("--base-url", default="http://127.0.0.1:8069/v1")
    p.add_argument("--max-images", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--thinking", action="store_true", help="enable Qwen thinking; otherwise disabled")
    p.add_argument("--claim-batch-size", type=int, default=6, help="same-prefix claims per request; 1 disables batching")
    p.add_argument("--claim-images", choices=("all", "cited"), default="cited",
                   help="claim attachments; every cited image is preserved")
    p.add_argument("--claim-observations", choices=("compact", "detailed"), default="compact",
                   help="compact stores claim facts in reason; independent blind keeps all image observations")
    p.set_defaults(function=prepare)
    p = sub.add_parser("run", help="explicitly execute prepared Qwen image review tasks")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-calls", type=int, required=True, help="total run-wide cap, including previous calls and retries")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--workers", type=int, default=4, help="parallel independent cases; each case remains chronological")
    p.set_defaults(function=run)
    p = sub.add_parser("inspect", help="offline integrity and task/call counts")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(function=inspect_run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    def handle_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_term)
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(1)
