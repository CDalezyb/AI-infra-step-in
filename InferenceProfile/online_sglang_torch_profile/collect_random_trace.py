#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(base_url: str, endpoint: str, body: dict[str, Any], timeout: float | None):
    url = base_url.rstrip("/") + endpoint
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        kwargs = {} if timeout is None else {"timeout": timeout}
        with urllib.request.urlopen(req, **kwargs) as resp:
            return resp.read().decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc


def check_server(base_url: str):
    url = base_url.rstrip("/") + "/server_info"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to SGLang server at {base_url}: {exc}") from exc


def start_profile(args: argparse.Namespace, trace_dir: Path):
    body = {
        "output_dir": str(trace_dir),
        "activities": args.profile_activities,
        "with_stack": args.with_stack,
        "record_shapes": args.record_shapes,
        "merge_profiles": bool(args.merge_profiles),
    }
    if args.profile_prefix:
        body["profile_prefix"] = args.profile_prefix

    print(post_json(args.base_url, "/start_profile", body, timeout=30))


def stop_profile(base_url: str):
    print(post_json(base_url, "/stop_profile", {}, timeout=None))


def run_random_benchmark(args: argparse.Namespace, script_dir: Path, trace_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        args.backend,
        "--model",
        args.model_path,
        "--base-url",
        args.base_url,
        "--dataset-name",
        args.dataset_name,
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        str(args.seed),
        "--disable-tqdm",
        "--warmup-requests",
        str(args.warmup_requests),
    ]

    if args.tokenize_prompt:
        cmd.append("--tokenize-prompt")
    if args.max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(args.max_concurrency)])
    if args.extra_request_body is not None:
        json.loads(args.extra_request_body)
        cmd.extend(["--extra-request-body", args.extra_request_body])

    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = str(trace_dir)

    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=script_dir, env=env, check=True)


def infer_profile_id(trace_dir: Path) -> str:
    sources = sorted(trace_dir.glob("*-TP-*.trace.json.gz"))
    if not sources:
        raise RuntimeError(f"No TP rank trace files found in {trace_dir}")

    profile_groups: dict[str, list[Path]] = {}
    for path in sources:
        match = re.match(r"(.+)-TP-\d+.*\.trace\.json\.gz$", path.name)
        if match:
            profile_groups.setdefault(match.group(1), []).append(path)

    if not profile_groups:
        raise RuntimeError(f"Cannot infer profile id from trace files in {trace_dir}")

    return max(profile_groups, key=lambda key: len(profile_groups[key]))


def ensure_merged_trace(trace_dir: Path):
    merged = sorted(glob.glob(str(trace_dir / "merged-*.trace.json.gz")))
    if merged:
        print(f"Merged aligned trace: {merged[-1]}")
        return

    profile_id = infer_profile_id(trace_dir)

    from sglang.srt.utils.profile_merger import ProfileMerger

    merger = ProfileMerger(str(trace_dir), profile_id)
    merged_path = merger.merge_chrome_traces()
    summary = merger.get_merge_summary()

    print(f"Merged aligned trace: {merged_path}")
    print(f"Source TP trace files: {summary.get('total_files', 'unknown')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect SGLang torch profiler traces with a random dataset "
            "and merge TP rank traces."
        )
    )
    parser.add_argument("--model-path", default="/data/public_models/Qwen3-8B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--trace-root", default="./sglang_trace")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument(
        "--dataset-name",
        default="random",
        choices=["random", "random-ids"],
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--random-input-len", type=int, default=6000)
    parser.add_argument("--random-output-len", type=int, default=25)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--tokenize-prompt", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extra-request-body", default=None)

    parser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "MEM", "CUDA_PROFILER", "RPD"],
    )
    parser.add_argument(
        "--with-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--merge-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profile-prefix", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        run_name = args.run_name or time.strftime("random_%Y%m%d_%H%M%S")
        trace_dir = Path(args.trace_root) / run_name
    if not trace_dir.is_absolute():
        trace_dir = script_dir / trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = trace_dir.resolve()

    print(f"Model: {args.model_path}")
    print(f"Base URL: {args.base_url}")
    print(f"Trace dir: {trace_dir}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Random input/output len: {args.random_input_len}/{args.random_output_len}")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Warmup requests: {args.warmup_requests}")
    print(f"Profile activities: {' '.join(args.profile_activities)}")
    print(f"Profile with stack: {args.with_stack}")
    print(f"Profile record shapes: {args.record_shapes}")
    print(f"Merge profiles: {args.merge_profiles}")

    profile_started = False
    check_server(args.base_url)
    try:
        start_profile(args, trace_dir)
        profile_started = True
        run_random_benchmark(args, script_dir, trace_dir)
    finally:
        if profile_started:
            try:
                stop_profile(args.base_url)
            except Exception as exc:
                print(f"Failed to stop profiler cleanly: {exc}", file=sys.stderr)

    ensure_merged_trace(trace_dir)


if __name__ == "__main__":
    main()
