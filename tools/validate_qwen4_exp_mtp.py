# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compare Qwen4Exp base, eager-draft and graph-draft runs on an NPU host.

Each mode owns an offline LLM in a separate subprocess. No existing server is
stopped. Engine arguments are read from a JSON object, using the Python LLM
argument names. Results include exact token IDs and batch-level speculative
counter deltas. This tool does not collect per-request acceptance metrics.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODES = ("base", "draft_eager", "draft_graph")
MODES = DEFAULT_MODES + ("prefill_graph", "decode_graph")
DIAGNOSTIC_WORKER = "tools.qwen4_exp_mtp_diagnostics_worker.DiagnosticNPUWorker"
COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}
PROMPTS = (
    "A shop has 18 apples and sells 7. It then receives 12. How many apples are there? Explain your steps.",
    "Calculate the sum of the first 30 positive integers. Show the reasoning.",
    "A train travels 180 kilometers in 3 hours. How far does it travel in 5 hours at the same speed?",
    "Explain why the product of two odd integers is odd, using algebra and an example.",
    "Write a Python function to count the frequency of words in a string, and explain its complexity.",
    "Compare breadth-first search and depth-first search with one example for each.",
    "Explain how rain forms, then describe the water cycle in five steps.",
    "A rectangle has perimeter 42 and length 13. Find its width and area. Explain each calculation.",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_engine_args(config, mode):
    kwargs = copy.deepcopy(config["engine_args"])
    kwargs.update(model=config["model"], seed=config["seed"], disable_log_stats=False)
    kwargs.setdefault("max_num_seqs", max(config["batch_sizes"]))
    kwargs.setdefault("enable_prefix_caching", False)
    kwargs.setdefault("async_scheduling", True)
    if kwargs.get("enforce_eager", False):
        raise ValueError("Target enforce_eager must be false to isolate draft graph behavior")
    compilation = kwargs.setdefault("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError("compilation_config must be a JSON object")
    compilation.setdefault("cudagraph_mode", "FULL_DECODE_ONLY")
    compilation.setdefault("cudagraph_capture_sizes", [1, 2, 4, 8, 12, 16, 24, 32, 64, 96, 128])
    if config.get("separate_graph_pools") and compilation["cudagraph_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("Separate graph pools require FULL_DECODE_ONLY to isolate the three full-graph managers")
    if mode == "base":
        kwargs.pop("speculative_config", None)
    else:
        speculative = dict(kwargs.get("speculative_config") or {})
        speculative.update(
            method="qwen3_5_mtp",
            num_speculative_tokens=config["num_speculative_tokens"],
            enforce_eager=mode == "draft_eager",
        )
        if config.get("disable_index_sharing"):
            speculative["index_share_for_mtp_iteration"] = False
        kwargs["speculative_config"] = speculative
    trace_limit = config.get("graph_trace_limit", 0)
    if mode in ("prefill_graph", "decode_graph") or config.get("separate_graph_pools") or trace_limit:
        if kwargs.get("worker_cls", "auto") != "auto":
            raise ValueError("The diagnostic worker cannot be combined with a custom worker_cls")
        kwargs["worker_cls"] = DIAGNOSTIC_WORKER
        overrides = kwargs.setdefault("hf_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("hf_overrides must be a JSON object for the diagnostic worker")
        # AscendConfig rejects unknown additional_config keys. ModelConfig's
        # dictionary HF overrides explicitly setattr new dict-valued fields.
        overrides["qwen4_exp_mtp_diagnostics"] = {
            "disable_prefill_graph": mode == "decode_graph",
            "disable_decode_graph": mode == "prefill_graph",
            "separate_graph_pools": bool(config.get("separate_graph_pools")),
            "trace_limit": trace_limit,
            "trace_dir": str(Path(config["output_dir"]) / "graph_traces" / mode),
        }
    return kwargs


def read_counters(metrics):
    values = dict.fromkeys(COUNTERS, None)
    positions = {}
    for metric in metrics:
        for key, name in COUNTERS.items():
            if metric.name == name:
                values[key] = (values[key] or 0) + float(metric.value)
        if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            position = str(metric.labels["position"])
            positions[position] = positions.get(position, 0.0) + float(metric.value)
    values["accepted_per_position"] = positions
    return values


def counter_delta(before, after):
    values = {
        key: after[key] - before[key] if after[key] is not None and before[key] is not None else None
        for key in COUNTERS
    }
    values["accepted_per_position"] = {
        position: value - before["accepted_per_position"].get(position, 0)
        for position, value in after["accepted_per_position"].items()
    }
    drafts = values["drafts"]
    accepted = values["accepted_tokens"]
    values["mean_acceptance_length"] = 1 + accepted / drafts if drafts and accepted is not None else None
    return values


def coverage_estimate(kwargs, batch_size, num_speculative_tokens):
    """Predict pure-decode token coverage; this is not observed graph dispatch."""
    width = num_speculative_tokens + 1
    compilation = kwargs["compilation_config"]
    sizes = compilation.get("cudagraph_capture_sizes", [])
    maximum = compilation.get("max_cudagraph_capture_size") or max(sizes, default=0)
    candidates = sorted(
        {
            math.ceil(size / width) * width
            for size in sizes
            if math.ceil(size / width) * width <= min(maximum, kwargs["max_num_seqs"] * width)
        }
    )
    required = batch_size * width
    return {
        "kind": "estimate_only",
        "target_verify_tokens": required,
        "target_capture_tokens": next((size for size in candidates if size >= required), None),
        "draft_decode_tokens": batch_size if num_speculative_tokens else None,
        "capture_sizes": sizes,
    }


def prepare_prompts(llm, prompts, count):
    tokenizer = llm.get_tokenizer()
    result = []
    for index in range(count):
        text = prompts[index % len(prompts)]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
        )
        result.append({"prompt_token_ids": ids})
    return result


def run_batch(llm, config, kwargs, mode, batch_size, run_index):
    from vllm import SamplingParams

    prompts = prepare_prompts(llm, config["prompts"], batch_size)
    limits = [max(8, config["max_tokens"] // (1 + index % 3)) for index in range(batch_size)]
    sampling = [SamplingParams(temperature=0, max_tokens=limit, ignore_eos=True) for limit in limits]
    before = read_counters(llm.get_metrics())
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - start
    after = read_counters(llm.get_metrics())
    requests = []
    for index, output in enumerate(outputs):
        metrics = output.metrics
        latency = None
        if metrics is not None and metrics.finished_time is not None:
            latency = metrics.finished_time - metrics.arrival_time
        requests.append(
            {
                "prompt_index": index % len(config["prompts"]),
                "max_tokens": limits[index],
                "prompt_token_ids": list(output.prompt_token_ids),
                "output_token_ids": list(output.outputs[0].token_ids),
                "latency_seconds": latency,
            }
        )
    num_spec = config["num_speculative_tokens"] if mode != "base" else 0
    return {
        "run_index": run_index,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(len(row["output_token_ids"]) for row in requests) / elapsed,
        "speculative_counters": counter_delta(before, after),
        "graph_coverage": coverage_estimate(kwargs, batch_size, num_spec),
        "requests": requests,
    }


def run_worker(config_path, mode):
    # Spawned TP workers inherit sys.path and must resolve the dotted worker
    # class from this checkout, including when the script is run by filename.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vllm import LLM, SamplingParams

    config = read_json(config_path)
    kwargs = build_engine_args(config, mode)
    llm = LLM(**kwargs)
    try:
        warmup = prepare_prompts(llm, config["prompts"], max(config["batch_sizes"]))
        llm.generate(warmup, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
        rows = []
        for _ in range(config["repeats"]):
            for batch_size in config["batch_sizes"]:
                row = run_batch(llm, config, kwargs, mode, batch_size, len(rows))
                rows.append(row)
                write_json(Path(config["output_dir"]) / f"{mode}.json", {"mode": mode, "runs": rows})
    finally:
        # This engine belongs to this child only; existing serving processes are untouched.
        llm.llm_engine.engine_core.shutdown()


def first_difference(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    return None if len(expected) == len(actual) else min(len(expected), len(actual))


def compare_results(results):
    mismatches = []
    repeats = []
    for mode, report in results.items():
        previous = {}
        for run in report["runs"]:
            baseline = next(
                (row for row in results.get("base", {}).get("runs", []) if row["run_index"] == run["run_index"]), None
            )
            for index, row in enumerate(run["requests"]):
                difference = (
                    first_difference(baseline["requests"][index]["output_token_ids"], row["output_token_ids"])
                    if baseline is not None
                    else None
                )
                key = (run["batch_size"], index, row["max_tokens"])
                repeated_difference = first_difference(
                    previous.get(key, row["output_token_ids"]), row["output_token_ids"]
                )
                entry = {"mode": mode, "run_index": run["run_index"], "request_index": index}
                if difference is not None:
                    mismatches.append(dict(entry, first_different_token=difference))
                if repeated_difference is not None:
                    repeats.append(dict(entry, first_different_token=repeated_difference))
                previous[key] = row["output_token_ids"]
    acceptance = {}
    for mode in results:
        if mode == "base":
            continue
        counters = [row["speculative_counters"] for row in results[mode]["runs"]]
        known = all(row["drafts"] is not None and row["accepted_tokens"] is not None for row in counters)
        drafts = sum(row["drafts"] for row in counters) if known else None
        accepted = sum(row["accepted_tokens"] for row in counters) if known else None
        acceptance[mode] = {
            "counter_status": "available" if known else "unknown",
            "drafts": drafts,
            "mean_acceptance_length": 1 + accepted / drafts if drafts else None,
        }
    eager = acceptance.get("draft_eager", {}).get("mean_acceptance_length")
    graph = acceptance.get("draft_graph", {}).get("mean_acceptance_length")
    return {
        "token_mismatches_against_base": mismatches,
        "repeated_run_token_mismatches": repeats,
        "acceptance": acceptance,
        "graph_to_eager_acceptance_ratio": graph / eager if graph is not None and eager else None,
        "notes": [
            "Base token comparison was run." if "base" in results else "Base mode absent: no base token comparison.",
            "Acceptance counters and MAL are batch aggregates, not per-request metrics.",
            "Graph coverage is a pure-decode estimate; use runtime logs or a trace to verify actual dispatch.",
            "Different output limits within a batch exercise shrinking batches and graph bucket transitions.",
            "Timing includes prefill and generation, excludes engine startup and the initial warmup.",
            "This zero-shot token diagnostic does not reproduce AISBench TPOT; graph tracing also adds host overhead.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--engine-args", help="JSON file containing LLM constructor keyword arguments")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompts", help="JSON file containing a nonempty list of user prompt strings")
    prompt_group.add_argument("--gsm8k-jsonl", help="GSM8K JSONL file; use each row's question as a zero-shot prompt")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many GSM8K rows")
    parser.add_argument("--output-dir", default="qwen4-exp-mtp-validation")
    parser.add_argument("--batch-sizes", default="1,2,8,3,8,2")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat", "--repeats", dest="repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes: " + ",".join(MODES))
    parser.add_argument(
        "--separate-graph-pools", action="store_true", help="One pool per target/draft-prefill/draft-decode manager"
    )
    parser.add_argument(
        "--disable-index-sharing", action="store_true", help="Disable sharing MTP top-k indices between steps"
    )
    parser.add_argument(
        "--graph-trace-limit",
        type=int,
        default=16,
        help="Maximum events per phase/event type per rank; 0 disables trace writes",
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail on token mismatches, unknown draft counters or acceptance regression"
    )
    parser.add_argument(
        "--min-acceptance-ratio", type=float, default=0.9, help="Minimum graph/eager MAL ratio for --check"
    )
    parser.add_argument("--dry-run", action="store_true", help="Write mode configurations without loading a model")
    parser.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") not in (None, "1"):
        raise SystemExit("VLLM_USE_V2_MODEL_RUNNER must be 1 for this MRV2 comparison")
    if args.worker_mode:
        run_worker(args.config, args.worker_mode)
        return 0
    if not args.model:
        raise SystemExit("--model is required")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if min(batch_sizes + [args.repeats, args.num_speculative_tokens]) < 1 or args.max_tokens < 8:
        raise SystemExit("Batch sizes, repeats and speculative tokens must be positive; max-tokens must be >= 8")
    if not 0 < args.min_acceptance_ratio <= 1:
        raise SystemExit("--min-acceptance-ratio must be greater than 0 and at most 1")
    modes = list(dict.fromkeys(args.modes.split(",")))
    if not modes or any(mode not in MODES for mode in modes):
        raise SystemExit("--modes must select from " + ",".join(MODES))
    if args.offset < 0 or args.graph_trace_limit < 0:
        raise SystemExit("--offset and --graph-trace-limit must be nonnegative")
    if args.gsm8k_jsonl:
        rows = [
            json.loads(line) for line in Path(args.gsm8k_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        prompts = [row["question"] for row in rows[args.offset :]]
    else:
        prompts = read_json(args.prompts) if args.prompts else list(PROMPTS)
    if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise SystemExit("--prompts must contain a nonempty JSON list of strings")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "engine_args": read_json(args.engine_args) if args.engine_args else {},
        "batch_sizes": batch_sizes,
        "prompts": prompts,
        "output_dir": str(output_dir),
        "modes": modes,
    }
    config_path = output_dir / "config.json"
    write_json(config_path, config)
    mode_args = {mode: build_engine_args(config, mode) for mode in modes}
    write_json(output_dir / "engine_args_by_mode.json", mode_args)
    if args.dry_run:
        print(f"Wrote configurations to {output_dir}")
        return 0
    environment = dict(os.environ)
    environment.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    results = {}
    for mode in modes:
        print(f"Running {mode}; log: {output_dir / (mode + '.log')}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", mode, "--config", str(config_path)]
        with (output_dir / f"{mode}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[mode] = read_json(output_dir / f"{mode}.json")
    summary = compare_results(results)
    ratio = summary["graph_to_eager_acceptance_ratio"]
    summary["min_acceptance_ratio"] = args.min_acceptance_ratio
    summary["acceptance_ratio_passed"] = ratio >= args.min_acceptance_ratio if ratio is not None else None
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = summary["token_mismatches_against_base"] or summary["repeated_run_token_mismatches"]
    failed = failed or any(row["drafts"] is None or row["drafts"] <= 0 for row in summary["acceptance"].values())
    failed = failed or summary["acceptance_ratio_passed"] is False
    return int(args.check and bool(failed))


if __name__ == "__main__":
    sys.exit(main())
