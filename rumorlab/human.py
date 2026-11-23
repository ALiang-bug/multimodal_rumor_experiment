"""Import externally supplied, version-bound human submissions."""

import base64
import json
from pathlib import Path

from .snapshot import snapshot_records
from .qa import project_options
from .requests import visible_material
from .validation import ID_FIELDS, canonical_hash, validate_rows


def blind_target_id(run_id, record_id):
    return "blind_" + canonical_hash([run_id, record_id])


def blind_review_tasks(state, contract, run_id, target_ids, guideline_version):
    if not guideline_version.strip():
        raise ValueError("Guideline version is required")
    records = snapshot_records(state, run_id, gold_votes=contract.gold_votes)
    source = state.db.execute("SELECT media_root FROM imports WHERE run_id = ?", (run_id,)).fetchone()
    root = Path(source["media_root"])
    samples = {row["sample_id"]: row for row in records if row["record_type"] == "sample"}
    evidence = [row for row in records if row["record_type"] == "evidence"]
    candidates = {row[ID_FIELDS[row["record_type"]]]: row for row in records
                  if row["record_type"] in {"annotation", "qa"} and row["status"] in
                  {"VALIDATED", "GENERATED", "REVIEWED", "ACCEPTED_SILVER", "HUMAN_PENDING"}}
    targets = sorted(set(target_ids)) if target_ids else sorted(
        rid for rid, row in candidates.items() if row["status"] == "HUMAN_PENDING"
    )

    def material(sample, *, media_ids=None, evidence_ids=None):
        if sample["status"] != "VALIDATED":
            raise ValueError("Blind review requires a valid sample")
        selected = {**sample}
        if media_ids is not None:
            selected["media"] = [asset for asset in sample["media"] if asset["media_id"] in media_ids]
            if {asset["media_id"] for asset in selected["media"]} != set(media_ids):
                raise ValueError("Blind review has unresolved QA media references")
        if evidence_ids is not None:
            selected["evidence_ids"] = evidence_ids
        visible, images = visible_material(selected, evidence, root, solver=True)
        for asset, raw in zip(visible["images"], images):
            asset["base64"] = base64.b64encode(raw).decode("ascii")
        return visible

    for rid in targets:
        if rid not in candidates:
            raise ValueError(f"Unavailable blind review target: {rid}")
        target = candidates[rid]
        sample = samples[target["sample_id"]]
        kind = target["record_type"]
        payload = target["payload"]
        task_id = payload["task_id"]
        task = {
            "target_id": blind_target_id(run_id, rid), "target_record_type": kind,
            "target_content_hash": target["content_hash"],
            "guideline_version": guideline_version, "is_mock": target["is_mock"], "task_id": task_id,
        }
        if kind == "annotation":
            task["materials"] = material(sample)
            task["target_claim"] = payload["target_claim"]
            task["task_definition"] = contract.labels["tasks"][task_id]
        else:
            solver = target["solver_input"]
            task["materials"] = material(sample, media_ids=solver["media_ids"], evidence_ids=solver["evidence_ids"])
            task["materials"]["text"] = solver["text"]
            if solver["reference_sample_ids"]:
                task["materials"]["references"] = [
                    material(samples[ref], evidence_ids=[]) for ref in solver["reference_sample_ids"]
                ]
            task["question"] = payload["question"]
            task["question_type"] = payload["question_type"]
            task["options"], _ = project_options(payload, seed=contract.experiment["seed"], qa_id=rid)
        yield task


def import_human(state, contract, run_id, submissions):
    with state.db:
        state.db.execute("BEGIN IMMEDIATE")
        records = snapshot_records(state, run_id, gold_votes=contract.gold_votes)
        source = state.db.execute("SELECT media_root FROM imports WHERE run_id = ?", (run_id,)).fetchone()
        by_id = {row[ID_FIELDS[row["record_type"]]]: row for row in records}
        blind_ids = {blind_target_id(run_id, rid): rid for rid, row in by_id.items()
                     if row["record_type"] in {"annotation", "qa"}}
        votes = {(row["target_id"], row["target_content_hash"], row["annotator_id"], row["vote_mode"])
                 for row in records if row["record_type"] == "human_vote" and row["status"] == "SUBMITTED"}
        accepted = []
        corrections = []
        for line, row in submissions:
            kind = row.get("record_type") if isinstance(row, dict) else None
            if kind not in {"human_vote", "adjudication"}:
                raise ValueError(f"Line {line}: expected human_vote or adjudication")
            if kind == "human_vote" and row.get("target_id") in blind_ids:
                row = {**row, "target_id": blind_ids[row["target_id"]]}
                if isinstance(row.get("qa_answer"), dict) and "letters" in row["qa_answer"]:
                    target = by_id[row["target_id"]]
                    if target["record_type"] != "qa":
                        raise ValueError(f"Line {line}: displayed options require a QA target")
                    _, mapping = project_options(
                        target["payload"], seed=contract.experiment["seed"], qa_id=row["target_id"],
                    )
                    answer = dict(row["qa_answer"])
                    letters = answer.pop("letters")
                    if (not isinstance(letters, list) or not all(isinstance(letter, str) for letter in letters)
                            or len(letters) != len(set(letters)) or not set(letters) <= mapping.keys()
                            or "answer_option_ids" in answer):
                        raise ValueError(f"Line {line}: invalid displayed option letters")
                    answer["answer_option_ids"] = [mapping[letter] for letter in letters]
                    row["qa_answer"] = answer
            errors = list(contract.validator(kind).iter_errors(row))
            if errors:
                raise ValueError(f"Line {line}: {errors[0].message}")
            rid = row[ID_FIELDS[kind]]
            if rid in by_id:
                if row != by_id[rid]:
                    raise ValueError(f"Line {line}: submission ID already has different content: {rid}")
                continue
            target = by_id.get(row["target_id"])
            if (not target or target["record_type"] != row["target_record_type"]
                    or target["content_hash"] != row["target_content_hash"]
                    or target["status"] in {"STALE", "EXCLUDED", "SKIPPED"}):
                raise ValueError(f"Line {line}: stale or invalid target version")
            if row["is_mock"] != target["is_mock"]:
                raise ValueError(f"Line {line}: human submission and target mock identity differ")
            if kind == "human_vote" and row["status"] == "SUBMITTED":
                key = (row["target_id"], row["target_content_hash"], row["annotator_id"], row["vote_mode"])
                if key in votes:
                    raise ValueError(f"Line {line}: duplicate vote for annotator and target version")
                votes.add(key)
                if row["annotation_answer"]:
                    answer = row["annotation_answer"]
                    if (answer["task_id"] != target["payload"]["task_id"]
                            or answer["target_claim"] != target["payload"]["target_claim"]):
                        raise ValueError(f"Line {line}: human answer binds the wrong task or claim")
                else:
                    answer = row["qa_answer"]
                    allowed_ids = {option["option_id"] for option in target["payload"]["options"]}
                    if not set(answer["answer_option_ids"]) <= allowed_ids:
                        raise ValueError(f"Line {line}: human QA answer cites unknown option")
                if not set(answer["evidence_ids"]) <= set(target["available_evidence_ids"]):
                    raise ValueError(f"Line {line}: human answer cites evidence outside target")
            by_id[rid] = row
            accepted.append((run_id, rid, json.dumps(row, ensure_ascii=False)))
            if kind == "adjudication" and row["status"] == "FINALIZED" and row["outcome"] == "CORRECTED":
                corrections.append((row, target))
        errors, _ = validate_rows(
            list(enumerate(by_id.values(), 1)), contract.schema, contract.labels,
            contract.gold_votes, Path(source["media_root"]),
        )
        if errors:
            raise ValueError("\n".join(errors))
        state.db.executemany("INSERT INTO human_submissions VALUES (?, ?, ?)", accepted)
        corrected_targets = set()
        for ruling, target in corrections:
            kind = target["record_type"]
            id_field = ID_FIELDS[kind]
            old_id = target[id_field]
            if old_id in corrected_targets:
                raise ValueError(f"Multiple finalized corrections for {old_id}")
            corrected_targets.add(old_id)
            payload = ruling[f"final_{kind}_payload"]
            if payload["task_id"] != target["payload"]["task_id"] or payload == target["payload"]:
                raise ValueError("Corrected payload must change content while preserving the task")
            if kind == "annotation" and payload["target_claim"] != target["payload"]["target_claim"]:
                raise ValueError("Corrected annotation must retain its target claim")
            digest = canonical_hash(payload)
            version_id = f"{kind}_" + canonical_hash([old_id, ruling["adjudication_id"], digest])
            revised = {
                **target, id_field: version_id, "payload": payload, "content_hash": digest,
                "input_hash": canonical_hash([target["input_hash"], ruling["content_hash"]]),
                "created_at": ruling["created_at"], "supersedes_id": old_id,
                "record_version": target["record_version"] + 1,
                "status": "SKIPPED" if payload.get("applicability") == "not_applicable" else "VALIDATED",
                "quality_tier": "unresolved", "review_ids": [], "human_vote_ids": ruling["human_vote_ids"],
                "adjudication_id": ruling["adjudication_id"],
                "producer": {"kind": "human", "role": "human", "model_id": None, "model_revision": None,
                             "run_id": target["producer"]["run_id"], "prompt_version": None, "prompt_hash": None},
            }
            state.db.execute("INSERT INTO human_submissions VALUES (?, ?, ?)",
                             (run_id, version_id, json.dumps(revised, ensure_ascii=False)))
            state._invalidate_in_transaction(old_id, target["content_hash"])
        if corrections:
            projected = snapshot_records(state, run_id, gold_votes=contract.gold_votes)
            errors, _ = validate_rows(
                list(enumerate(projected, 1)), contract.schema, contract.labels,
                contract.gold_votes, Path(source["media_root"]),
            )
            if errors:
                raise ValueError("\n".join(errors))
    return len(accepted)
