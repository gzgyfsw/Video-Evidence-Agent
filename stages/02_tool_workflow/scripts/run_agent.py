#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from agent.core import AgentConfig, RuleBasedVideoSafetyAgent
from agent.mock import MockClassifier, MockLocator
from agent.qwen_backend import QwenTokenLikelihoodClassifier
from agent.tools import MotionEventLocator, RiskSemanticLocator, TemporalEvidenceExpander
from agent.policy_agent import PolicyAgentConfig, ToolCallingVideoSafetyAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the adaptive video-safety agent")
    parser.add_argument("video")
    parser.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--model-path", help="required for --backend qwen")
    parser.add_argument("--locator-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--locator-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--locator-candidate-frames", type=int, default=0,
                        help="0 selects duration-adaptive 2-second sampling")
    parser.add_argument("--mock-locator", action="store_true", help="avoid downloading/loading SigLIP")
    parser.add_argument("--initial-frames", type=int, default=0,
                        help="0 selects duration-adaptive 4/8/12/16 initial frames")
    parser.add_argument("--expansion-frames", type=int, default=16)
    parser.add_argument("--max-total-frames", type=int, default=48)
    parser.add_argument("--max-frames-per-action", type=int, default=16)
    parser.add_argument("--mode", choices=["rule", "policy"], default="rule")
    parser.add_argument("--output", help="optional path for the JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "qwen":
        if not args.model_path:
            raise SystemExit("--model-path is required for --backend qwen")
        classifier = QwenTokenLikelihoodClassifier(args.model_path)
    else:
        classifier = MockClassifier()
    locator = MockLocator() if args.mock_locator else RiskSemanticLocator(
        model_name=args.locator_model, device=args.locator_device,
        candidate_frames=args.locator_candidate_frames or None,
    )
    if args.mode == "policy":
        if args.backend != "qwen":
            raise SystemExit("--mode policy currently requires --backend qwen")
        config = PolicyAgentConfig(
            initial_frames=args.initial_frames or None,
            default_action_frames=args.expansion_frames,
            max_total_frames=args.max_total_frames,
            max_frames_per_action=args.max_frames_per_action,
        )
        agent = ToolCallingVideoSafetyAgent(
            classifier, locator, TemporalEvidenceExpander(), config,
            motion_locator=MotionEventLocator(),
        )
    else:
        config = AgentConfig(initial_frames=args.initial_frames or 16, expansion_frames=args.expansion_frames)
        agent = RuleBasedVideoSafetyAgent(classifier, locator, TemporalEvidenceExpander(), config)
    payload = json.dumps(agent.run(args.video).to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
