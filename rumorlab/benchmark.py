"""Replay fixed real val annotation requests without writing workflow state."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from math import ceil
from pathlib import Path
import sqlite3
from statistics import median
import sys
from time import perf_counter

from .m1 import connect_role, load_lock
from .pilot import _inputs
from .requests import build_request
from .validation import canonical_hash


def _measure(client, requests, concurrency, seed):
    def send(request):
        result, _ = client.complete(request, seed=seed)
        if not result.schema_valid:
            return result, result.technical_error or f"invalid_response:{result.finish_reason}"
        if request.task_id != "reality_plausibility" and result.parsed_payload["applicability"] != "applicable":
            return result, "invalid_applicability"
        return result, None

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(send, requests))
    elapsed = perf_counter() - started
    failures = Counter(error for _, error in results if error is not None)
    valid = [result for result, error in results if error is None]
    latency = sorted(result.latency_ms for result in valid)
    tokens = sum(result.usage.get("completion_tokens") or 0 for result in valid)
    return {
        "concurrency": concurrency, "attempted": len(results), "succeeded": len(valid),
        "failed": len(results) - len(valid), "errors": dict(failures), "elapsed_seconds": elapsed,
        "successful_requests_per_minute": 60 * len(valid) / elapsed,
        "completion_tokens_per_second": tokens / elapsed,
        "latency_p50_ms": median(latency) if latency else None,
        "latency_p95_ms": latency[ceil(0.95 * len(latency)) - 1] if latency else None,
    }


def benchmark_annotator(contract, db_path, run_id, model_lock, levels, *, sample_count):
    if not levels or len(set(levels)) != len(levels) or any(not 1 <= level <= 32 for level in levels):
        raise ValueError("--levels must contain distinct concurrency levels between 1 and 32")
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        row, _, _, samples, evidence = _inputs(db, run_id)
    if not 1 <= sample_count <= len(samples):
        raise ValueError(f"--samples must be between 1 and {len(samples)} for this val run")
    seed = contract.experiment["seed"]
    sample_ids = sorted(samples, key=lambda sid: canonical_hash([seed, sid]))[:sample_count]
    tasks = contract.experiment["annotation"]["model_tasks"]
    requests = [build_request(contract, "annotate", task, sample, sample["target_claims"][0],
                              evidence.values(), Path(row["media_root"]))
                for sample in (samples[sid] for sid in sample_ids) for task in tasks]
    if max(levels) > len(requests):
        raise ValueError("The val run does not contain enough fixed requests for the requested concurrency")
    total_calls = len(levels) * len(requests) + len(tasks)
    runtime = contract.experiment["runtime"]
    budget = contract.models["models"]["annotator"]["generation"]["max_output_tokens"]
    if total_calls > runtime["max_calls_per_run"] or total_calls * budget > runtime["max_generated_tokens_per_run"]:
        raise ValueError("Benchmark exceeds the configured request budget")
    lock, lock_digest = load_lock(contract, "annotator", model_lock)
    client = connect_role(contract, "annotator")
    warmup = _measure(client, requests[:len(tasks)], min(4, max(levels)), seed)
    if warmup["failed"]:
        raise ValueError(f"Benchmark warm-up failed: {json.dumps(warmup['errors'])}")
    results = []
    for level in levels:
        print(f"Benchmark concurrency={level}, requests={len(requests)}", file=sys.stderr, flush=True)
        measured = _measure(client, requests, level, seed)
        results.append(measured)
        print(json.dumps(measured), file=sys.stderr, flush=True)
    winner = max((result for result in results if not result["failed"]),
                 key=lambda result: (result["successful_requests_per_minute"], -result["concurrency"]), default=None)
    return {
        "model_id": lock["model_id"], "model_revision": lock["revision"], "model_lock_sha256": lock_digest,
        "run_id": run_id, "seed": seed, "sample_ids": sample_ids,
        "workload_sha256": canonical_hash([seed, [request.request_id for request in requests]]),
        "warmup_requests": warmup["attempted"], "results": results,
        "best_concurrency": winner["concurrency"] if winner is not None else None,
    }
