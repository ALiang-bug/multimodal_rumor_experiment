"""Command-line entry points for the experiment."""

import argparse
import json
from pathlib import Path
import sys

import jsonschema
import yaml

from .config import Contract
from .validation import validate_rows


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc.msg}") from exc


def main():
    parser = argparse.ArgumentParser(prog="rumorlab")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Check the local configuration and schema")
    preflight.add_argument("--mode", choices=["mock", "real"], default="mock")
    preflight.add_argument("--role", choices=["annotator", "reviewer"])
    preflight.add_argument("--model-lock", type=Path)
    preflight.add_argument("--snapshot", type=Path, help="Record local deployment metadata before probing")
    validate = commands.add_parser("validate-data", help="Validate a closed-world JSONL snapshot")
    validate.add_argument("--records", type=Path)
    validate.add_argument("--data-root", type=Path)
    release = commands.add_parser("release", help="Check a candidate release snapshot")
    release.add_argument("--records", required=True, type=Path)
    release.add_argument("--data-root", type=Path)
    release.add_argument("--confirmed-links", type=Path)
    release.add_argument("--check-only", required=True, action="store_true")
    audit = commands.add_parser("audit", help="Sample a frozen reviewer population")
    audit.add_argument("--population", required=True, type=Path)
    evaluate = commands.add_parser("evaluate", help="Estimate reviewer metrics from audit judgments")
    evaluate.add_argument("--population", required=True, type=Path)
    evaluate.add_argument("--selections", required=True, type=Path)
    evaluate.add_argument("--judgments", required=True, type=Path)
    qa_eval = commands.add_parser("evaluate-qa", help="Score a fixed QA snapshot, including failed predictions")
    qa_eval.add_argument("--records", required=True, type=Path)
    qa_eval.add_argument("--predictions", required=True, type=Path)
    qa_eval.add_argument("--data-root", type=Path)
    qa_eval.add_argument("--mock", action="store_true")
    import_data = commands.add_parser("ingest", help="Import a source dataset or bundled mock snapshot")
    import_data.add_argument("--run-id", required=True)
    import_data.add_argument("--records", type=Path)
    import_data.add_argument("--data-root", type=Path)
    import_data.add_argument("--state-db", type=Path)
    import_kind = import_data.add_mutually_exclusive_group()
    import_kind.add_argument("--fixture", action="store_true")
    import_kind.add_argument("--adapter", choices=["generic_jsonl", "mmfakebench"])
    import_data.add_argument("--sample-size", type=int, help="MMFakeBench sample count (default: entire source split)")
    smoke = commands.add_parser("mock-run", help="Run the imported synthetic fixture through four model stages")
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--state-db", type=Path)
    smoke.add_argument("--script", type=Path, help="JSONL of sample, claim, task and scripted responses")
    smoke.add_argument("--provenance", action="store_true", help="Also derive source labels from the manifest")
    annotate = commands.add_parser("annotate", help="Annotate a bounded MMFakeBench val batch")
    annotate.add_argument("--offset", type=int, default=0, help="Number of samples to skip in the fixed val ordering")
    annotate.add_argument("--batch-offsets", type=int, nargs="+",
                          help="Share one inference queue across batches named <batch-id>-<offset>; --limit applies per batch")
    review_labels = commands.add_parser("review-labels", help="Review the saved annotations of a val batch")
    generate_qa = commands.add_parser("generate-qa", help="Generate QA from accepted val annotations")
    review_qa = commands.add_parser("review-qa", help="Review the saved QA candidates of a val batch")
    for command in (annotate, review_labels, generate_qa, review_qa):
        command.add_argument("--run-id", required=True, help="Imported val snapshot")
        command.add_argument("--batch-id", required=True, help="Versioned pilot batch within the snapshot")
        command.add_argument("--reviewer-lock", required=True, type=Path)
        command.add_argument("--state-db", type=Path)
        command.add_argument("--workers", type=int, help="Maximum concurrent model requests")
    for command, unit in ((annotate, "samples"), (generate_qa, "accepted labels")):
        command.add_argument("--limit", required=True, type=int, help=f"Maximum number of {unit}")
        command.add_argument("--annotator-lock", required=True, type=Path)
    generate_qa.add_argument("--label-batch-id", required=True, help="Completed source label batch")
    for command in (generate_qa, review_qa):
        command.add_argument("--dry-run", action="store_true", help="Report eligible work without requesting the model")
    benchmark = commands.add_parser("benchmark-vllm", help="Replay fixed val annotation requests at several concurrency levels")
    benchmark.add_argument("--run-id", required=True, help="Imported MMFakeBench val snapshot")
    benchmark.add_argument("--state-db", type=Path)
    benchmark.add_argument("--model-lock", required=True, type=Path)
    benchmark.add_argument("--samples", type=int, default=8, help="Number of distinct val samples (default: 8)")
    benchmark.add_argument("--repeats", type=int, default=1, help="Replays of the fixed request set per level")
    benchmark.add_argument("--timeout-seconds", type=int, help="Override request timeout for this benchmark only")
    benchmark.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32])
    invalidate = commands.add_parser("invalidate", help="Retire a version and invalidate its downstream jobs")
    invalidate.add_argument("--record-id", required=True)
    invalidate.add_argument("--content-hash", required=True)
    invalidate.add_argument("--state-db", type=Path)
    export = commands.add_parser("export-snapshot", help="Write the imported snapshot with current statuses to stdout")
    export.add_argument("--run-id", required=True)
    export.add_argument("--state-db", type=Path)
    human = commands.add_parser("import-human", help="Import version-bound human review records")
    human.add_argument("--run-id", required=True)
    human.add_argument("--records", required=True, type=Path)
    human.add_argument("--state-db", type=Path)
    blind = commands.add_parser("export-human", help="Export blind review assignments")
    blind.add_argument("--run-id", required=True)
    blind.add_argument("--guideline-version", required=True)
    blind.add_argument("--target-id", action="append", default=[])
    blind.add_argument("--state-db", type=Path)
    grouping_commands = []
    for name in ("group", "split"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--state-db", type=Path)
        command.add_argument("--confirmed-links", type=Path)
        grouping_commands.append(command)
    for command in (preflight, validate, release, audit, evaluate, qa_eval, import_data, smoke, annotate,
                    review_labels, generate_qa, review_qa, benchmark, invalidate, export, human, blind, *grouping_commands):
        command.add_argument("--config", default="configs/experiment.yaml", type=Path)
    args = parser.parse_args()
    try:
        contract = Contract(args.config)
        if args.command == "benchmark-vllm":
            from .benchmark import benchmark_annotator

            path = args.state_db or contract.root / contract.experiment["paths"]["state_db"]
            if not path.is_file():
                raise ValueError(f"Imported val state database not found: {path}")
            report = benchmark_annotator(contract, path, args.run_id, args.model_lock, args.levels,
                                         sample_count=args.samples, repeats=args.repeats, timeout=args.timeout_seconds)
            print(json.dumps(report, ensure_ascii=False))
            return 1 if report["best_concurrency"] is None else 0
        if args.command in {"annotate", "review-labels", "generate-qa", "review-qa"}:
            from . import pilot
            from .state import State

            workers = args.workers if args.workers is not None else contract.experiment["runtime"]["workers_initial"]
            if type(workers) is not int or workers < 1:
                raise ValueError("--workers must be a positive integer")
            state = State(args.state_db or contract.root / contract.experiment["paths"]["state_db"])
            try:
                batch = (state, contract, args.run_id, args.batch_id)
                if args.command == "annotate":
                    if args.batch_offsets is None:
                        report = pilot.annotate(*batch, args.limit, args.annotator_lock, args.reviewer_lock,
                                                offset=args.offset, workers=workers)
                    else:
                        if args.offset:
                            raise ValueError("Use --batch-offsets or --offset")
                        batches = [(f"{args.batch_id}-{offset}", offset, args.limit) for offset in args.batch_offsets]
                        reports = pilot.annotate_batches(*batch[:3], batches, args.annotator_lock, args.reviewer_lock,
                                                         workers=workers)
                        report = {"stage": "annotate", "batches": reports,
                                  "run_timings": reports[0]["run_timings"],
                                  "failed": sum(item["failed"] for item in reports)}
                elif args.command == "review-labels":
                    report = pilot.review_labels(*batch, args.reviewer_lock, workers=workers)
                elif args.command == "generate-qa":
                    report = pilot.generate_qa(*batch, args.label_batch_id, args.limit, args.annotator_lock,
                                               args.reviewer_lock, workers=workers, dry_run=args.dry_run)
                else:
                    report = pilot.review_qa(*batch, args.reviewer_lock, workers=workers, dry_run=args.dry_run)
            finally:
                state.close()
            print(json.dumps(report, ensure_ascii=False))
            return 1 if report["failed"] else 0
        if args.command in {"audit", "evaluate"}:
            from .audit import draw_audit, reviewer_metrics

            population = [row for _, row in read_jsonl(args.population)]
            policy = contract.experiment["audit"]
            if args.command == "audit":
                for selection in draw_audit(population, policy, seed=contract.experiment["seed"]):
                    print(json.dumps(selection))
            else:
                selections = [row for _, row in read_jsonl(args.selections)]
                judgments = [row for _, row in read_jsonl(args.judgments)]
                print(json.dumps(reviewer_metrics(population, selections, judgments, policy)))
            return 0
        if args.command == "preflight":
            if args.mode == "real":
                from .m1 import record_lock, run_smoke

                if not args.role or not args.model_lock:
                    raise ValueError("Real preflight requires --role and --model-lock")
                if args.snapshot:
                    record_lock(contract, args.role, args.snapshot, args.model_lock)
                report = run_smoke(contract, args.role, args.model_lock)
                print(json.dumps(report, ensure_ascii=False))
                return 0 if report["passed"] else 1
            print(json.dumps({
                "is_mock": True, "config_hash": contract.config_hash,
                "tasks": contract.experiment["annotation"]["tasks"],
            }))
            return 0
        if args.command == "ingest":
            from .ingest import import_snapshot, ingest
            from .state import State

            source = contract.experiment["source"]
            adapter = "fixture" if args.fixture else args.adapter or source["adapter"]
            if args.sample_size is not None and adapter != "mmfakebench":
                raise ValueError("--sample-size is only supported by the mmfakebench adapter")
            configured_path = contract.experiment["fixture"]["records"] if args.fixture else source["input_jsonl"]
            path = args.records or (contract.root / configured_path if configured_path else None)
            root = args.data_root or contract.root
            if not args.fixture and args.data_root is None and source["media_root"]:
                root = contract.root / source["media_root"]
            root = root.resolve()
            sampling = None
            if adapter == "mmfakebench":
                from .mmfakebench import prepare_mmfakebench

                if args.data_root is None and not source["media_root"]:
                    raise ValueError("Supply --data-root or configure source.media_root for MMFakeBench")
                if path is None:
                    path = root / "MMFakeBench_val.json"
                    if not path.is_file():
                        path = root / "MMFakeBench_val/source/MMFakeBench_val.json"
                rows, sampling = prepare_mmfakebench(path, root, contract.experiment["seed"], args.sample_size)
            elif adapter in {"generic_jsonl", "fixture"}:
                if path is None:
                    raise ValueError("Supply --records, configure source.input_jsonl or select --fixture")
                rows = list(read_jsonl(path))
            else:
                raise ValueError(f"Unsupported source adapter: {adapter!r}")
            state = State(args.state_db or contract.root / contract.experiment["paths"]["state_db"])
            try:
                if sampling is None:
                    count, inserted = ingest(state, rows, contract, root, args.run_id, fixture=args.fixture)
                else:
                    count, inserted = import_snapshot(state, rows, contract, root, args.run_id, sampling=sampling)
            finally:
                state.close()
            report = {"records": count, "inserted": inserted, "run_id": args.run_id}
            if sampling is not None:
                report["sampling"] = {key: value for key, value in sampling.items() if key not in {"selection", "quarantine"}}
                report["sampling"]["quarantined_rows"] = len(sampling["quarantine"])
            print(json.dumps(report))
            return 0
        if args.command in {"mock-run", "group", "split", "invalidate", "export-snapshot", "import-human", "export-human"}:
            from .grouping import build_groups, freeze_plan
            from .pipeline import mock_run
            from .state import State

            state = State(args.state_db or contract.root / contract.experiment["paths"]["state_db"])
            try:
                if args.command == "export-human":
                    from .human import blind_review_tasks

                    assignments = list(blind_review_tasks(
                        state, contract, args.run_id, args.target_id, args.guideline_version,
                    ))
                    for assignment in assignments:
                        print(json.dumps(assignment))
                    return 0
                if args.command == "import-human":
                    from .human import import_human

                    count = import_human(state, contract, args.run_id, list(read_jsonl(args.records)))
                    print(json.dumps({"inserted": count}))
                    return 0
                if args.command == "export-snapshot":
                    from .snapshot import snapshot_records

                    for record in snapshot_records(state, args.run_id, gold_votes=contract.gold_votes):
                        print(json.dumps(record))
                    return 0
                if args.command == "mock-run":
                    cases = [row for _, row in read_jsonl(args.script)] if args.script else None
                    report = mock_run(state, contract, args.run_id, cases=cases, provenance=args.provenance)
                elif args.command == "invalidate":
                    if len(args.content_hash) != 64 or any(char not in "0123456789abcdef" for char in args.content_hash):
                        raise ValueError("Expected a lowercase SHA-256 content hash")
                    report = {"stale_jobs": state.invalidate(args.record_id, args.content_hash)}
                else:
                    snapshot = state.db.execute("SELECT records FROM imports WHERE run_id = ?", (args.run_id,)).fetchone()
                    if snapshot is None:
                        raise ValueError(f"No imported snapshot for {args.run_id!r}")
                    links = [row for _, row in read_jsonl(args.confirmed_links)] if args.confirmed_links else None
                    records = json.loads(snapshot["records"])
                    if args.command == "split":
                        digest, report = freeze_plan(state, args.run_id, records, contract.experiment["splitting"], links)
                        report["split_manifest_id"] = "split_" + digest
                    else:
                        report = build_groups(records, links or ())
                print(json.dumps(report, ensure_ascii=False))
            finally:
                state.close()
            return 0
        path = args.records or contract.root / contract.experiment["fixture"]["records"]
        rows = list(read_jsonl(path))
        if args.command == "release":
            from .release import check_release

            links = [row for _, row in read_jsonl(args.confirmed_links)] if args.confirmed_links else []
            errors = check_release(rows, contract, args.data_root or contract.root, links)
            if errors:
                raise ValueError("\n".join(errors))
            print(f"Release gate passed for {len(rows)} records.")
            return 0
        errors, paths = validate_rows(
            rows, contract.schema, contract.labels, contract.gold_votes,
            args.data_root or contract.root,
        )
        if errors:
            raise ValueError("\n".join(errors))
        if args.command == "evaluate-qa":
            from .qa import score_snapshot

            predictions = [row for _, row in read_jsonl(args.predictions)]
            print(json.dumps(score_snapshot([row for _, row in rows], predictions,
                                            seed=contract.experiment["seed"], mock=args.mock)))
            return 0
        print(f"Validated {len(rows)} records and {len(paths)} media paths.")
        return 0
    except (OSError, ValueError, KeyError, jsonschema.SchemaError, jsonschema.ValidationError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
