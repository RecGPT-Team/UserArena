"""Run model evaluation or score saved UserArena rollout and replay traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics.l3 import SCHEME_LABELS
from . import report
from .report import run_rollout_evaluation
from .trace import read_replay_trace
from simulator.images import DEFAULT_IMAGE_HOSTS, ImageStore
from simulator.rollout import inspect_dataset, run_dataset


def build_generation_input(prompt, images, tokenizer):
    if prompt.count("<image>") != len(images):
        raise ValueError("Image placeholders do not match supplied images")
    messages = [{"role": "user", "content": prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if rendered.count("<|image_pad|>") != len(images):
        raise ValueError("Checkpoint chat template changed image placeholders")
    result = {"prompt": rendered}
    if images:
        result["multi_modal_data"] = {"image": images}
    return result


class VllmGenerator:
    def __init__(
        self,
        model_path,
        *,
        history_images=3,
        max_model_len=32768,
        max_tokens=512,
        image_resolution=200704,
        temperature=1.0,
        top_p=0.95,
        seed=42,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=False,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Install vLLM in a compatible inference environment; see README.md"
            ) from exc
        self.llm = LLM(
            model=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            seed=seed,
            limit_mm_per_prompt={"image": history_images + 1},
            mm_processor_kwargs={
                "min_pixels": image_resolution,
                "max_pixels": image_resolution,
            },
        )
        self.sampling = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.provenance = {
            "backend": "vllm",
            "temperature": temperature,
            "top_p": top_p,
            "generation_limit": max_tokens,
            "image_resolution": image_resolution,
            "seed": seed,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

    def __call__(self, prompt, images):
        inputs = build_generation_input(prompt, images, self.tokenizer)
        result = self.llm.generate([inputs], self.sampling, use_tqdm=False)[0]
        return (
            result.outputs[0].text,
            len(result.prompt_token_ids),
            len(result.outputs[0].token_ids),
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    run = commands.add_parser("run", help="Run a local vLLM checkpoint and score its trajectories")
    run.add_argument("--model-path", required=True)
    run.add_argument(
        "--image-root", type=Path, help="Defaults to the dataset directory"
    )
    run.add_argument("--image-cache", type=Path)
    run.add_argument(
        "--allowed-image-host", action="append", metavar="HOST",
        help="Allow an exact HTTPS image hostname; repeat for multiple hosts. "
             "Replaces the default allowlist (img.alicdn.com).",
    )
    run.add_argument("--cross-session", action="store_true")
    run.add_argument(
        "--max-actions",
        type=int,
        help="Executed-action budget: default 31; 42 with --cross-session",
    )
    run.add_argument(
        "--history-actions", type=int, help="Default 10; 20 with --cross-session"
    )
    run.add_argument(
        "--history-images", type=int, help="Default 3; 5 with --cross-session"
    )
    run.add_argument("--max-model-len", type=int, default=32768)
    run.add_argument("--max-tokens-per-step", type=int, default=512)
    run.add_argument("--image-resolution", type=int, default=200704)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--max-try", type=int, default=1)
    run.add_argument(
        "--retry-feedback", choices=("reason", "none"), default="reason"
    )
    run.add_argument(
        "--capture-cases",
        action="store_true",
        help="Save exact prompts and ordered viewport images locally",
    )
    rollout = commands.add_parser("rollout", help="Score saved rollout traces against logged references")
    replay = commands.add_parser("replay", help="Score saved per-state replay traces")
    for command in (run, rollout):
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--model-label", default="OPTD")
        command.add_argument("--min-cohort", type=int, default=30)
        command.add_argument("--cohort-schemes", nargs="+", choices=tuple(SCHEME_LABELS))
    for command in (run, rollout, replay):
        command.add_argument("--output-dir", required=True, type=Path)
    for command in (rollout, replay):
        command.add_argument("--traces", required=True, type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args, parser)
    except OSError as exc:
        parser.exit(
            2, f"evaluation: file or network operation failed ({type(exc).__name__}); "
            "check input files, output directory and access permissions.\n",
        )


def _execute(args, parser):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")
    if args.mode in ("run", "rollout"):
        args.model_label = args.model_label.strip()
        if not args.model_label:
            parser.error("model label must not be empty")
    if args.mode == "run":
        if args.max_actions is None:
            args.max_actions = 42 if args.cross_session else 31
        if args.history_actions is None:
            args.history_actions = 20 if args.cross_session else 10
        if args.history_images is None:
            args.history_images = 5 if args.cross_session else 3
        if min(args.history_actions, args.history_images) < 0:
            parser.error("History limits must be non-negative")
        if args.cross_session and args.history_actions != 20:
            parser.error("The cross-session prompt requires --history-actions 20")
        if (
            min(
                args.max_actions,
                args.max_try,
                args.max_model_len,
                args.max_tokens_per_step,
                args.image_resolution,
                args.min_cohort,
                args.tensor_parallel_size,
            )
            < 1
        ):
            parser.error(
                "Budgets, resolution, cohort size and parallel size must be positive"
            )
        if (
            args.temperature < 0
            or not 0 < args.top_p <= 1
            or not 0 < args.gpu_memory_utilization <= 1
        ):
            parser.error("Invalid sampling or GPU memory settings")
        try:
            audit = inspect_dataset(
                args.data,
                cross_session=args.cross_session,
                cohort_schemes=args.cohort_schemes,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"Dataset ready: {audit['sessions']} visits across {audit['users']} records",
            flush=True,
        )
        generate = VllmGenerator(
            args.model_path,
            history_images=args.history_images,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens_per_step,
            image_resolution=args.image_resolution,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
        )
        payload = run_dataset(
            args.data,
            args.output_dir,
            generate,
            image_store=ImageStore(
                args.image_root or args.data.parent, cache_dir=args.image_cache,
                allowed_hosts=(args.allowed_image_host
                               if args.allowed_image_host is not None else DEFAULT_IMAGE_HOSTS),
            ),
            tokenizer=generate.tokenizer,
            model_label=args.model_label,
            cross_session=args.cross_session,
            max_actions=args.max_actions,
            history_actions=args.history_actions,
            history_images=args.history_images,
            cutoff_len=args.max_model_len,
            max_try=args.max_try,
            retry_feedback=args.retry_feedback == "reason",
            capture_cases=args.capture_cases,
            min_cohort=args.min_cohort,
            cohort_schemes=args.cohort_schemes,
            provenance=generate.provenance,
        )
        print(
            f"Scored {payload['sim_episodes']} episodes; report.json saved in the output directory"
        )
        return
    if args.mode == "rollout":
        if args.min_cohort < 1:
            parser.error("--min-cohort must be positive")
        with args.traces.open(encoding="utf-8") as handle:
            traces = [json.loads(line) for line in handle if line.strip()]
        if any(trace.get("kind") != "rollout" for trace in traces):
            parser.error(
                "Expected canonical rollout records, including failed/empty episodes"
            )
        simulated = [trace for trace in traces if trace.get("source") == "sim"]
        payload = run_rollout_evaluation(
            simulated,
            args.data,
            args.output_dir,
            args.model_label,
            min_cohort=args.min_cohort,
            cohort_schemes=args.cohort_schemes,
        )
    else:
        with args.traces.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if "steps" in record or "gt_action" not in record:
                        parser.error(
                            "Replay requires one canonical ReplayStep per line, including gt_action"
                        )
        trace = read_replay_trace(args.traces)
        if not trace.steps:
            parser.error("Replay trace is empty")
        payload = report.replay_report(trace)
        report.write_report(args.output_dir / "report.json", payload)
        report.write_report(args.output_dir / "headline.json", payload["headline"])
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
