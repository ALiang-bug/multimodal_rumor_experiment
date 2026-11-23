"""Prepare display options and recover stable answers for scoring."""

import random
import re

from .validation import canonical_hash


def project_options(payload, *, seed, qa_id):
    question = payload.get("question", "")
    if re.search(r"(?:正确答案|标准答案|参考答案|correct answer|answer key)\s*[:：=]", question, re.IGNORECASE):
        raise ValueError("Question exposes an explicit answer marker")
    options = payload["options"]
    ids = {option["option_id"] for option in options}
    texts = {" ".join(option["text"].casefold().split()) for option in options}
    correct = set(payload["answer_option_ids"])
    if len(ids) != len(options) or len(texts) != len(options) or "" in texts:
        raise ValueError("Duplicate or empty QA option")
    if not correct or not correct <= ids:
        raise ValueError("Unknown or empty answer option")
    if payload["question_type"] == "single_choice" and len(correct) != 1:
        raise ValueError("Single-choice question requires exactly one answer")
    ordered = sorted(options, key=lambda option: option["option_id"])
    random.Random(canonical_hash([seed, qa_id])).shuffle(ordered)
    display = []
    mapping = {}
    for index, option in enumerate(ordered):
        letter = chr(ord("A") + index)
        display.append({"letter": letter, "text": option["text"]})
        mapping[letter] = option["option_id"]
    return display, mapping


def score_options(letters, mapping, answer_option_ids):
    if not isinstance(letters, list) or not all(isinstance(letter, str) for letter in letters):
        return False
    if len(letters) != len(set(letters)) or not set(letters) <= mapping.keys():
        return False
    return {mapping[letter] for letter in letters} == set(answer_option_ids)


def score_snapshot(records, predictions, *, seed, mock=False):
    samples = {row["sample_id"]: row for row in records if row["record_type"] == "sample"}
    questions = {row["qa_id"]: row for row in records if row["record_type"] == "qa"
                 and (row["is_mock"] if mock else (
                     not row["is_mock"] and row["status"] == "ACCEPTED_GOLD"
                     and row["quality_tier"] == "gold" and samples[row["sample_id"]]["split"] == "test"
                 )) and row["status"] not in {"STALE", "EXCLUDED", "HUMAN_PENDING"}}
    if not questions:
        raise ValueError("No eligible QA records in the evaluation snapshot")
    by_id = {}
    for row in predictions:
        rid = row["qa_id"]
        if rid not in questions or rid in by_id:
            raise ValueError(f"Duplicate or out-of-scope prediction: {rid}")
        if row["status"] not in {"answered", "invalid", "timeout", "refused", "technical_error"}:
            raise ValueError(f"Unknown prediction status for {rid}")
        by_id[rid] = row
    counts = dict.fromkeys(("correct", "incorrect", "invalid", "timeout", "refused", "technical_error", "missing"), 0)
    for rid, question in questions.items():
        prediction = by_id.get(rid)
        if prediction is None:
            counts["missing"] += 1
            continue
        status = prediction["status"]
        if status != "answered":
            counts[status] += 1
            continue
        payload = question["payload"]
        _, mapping = project_options(payload, seed=seed, qa_id=rid)
        letters = prediction.get("letters")
        if (not isinstance(letters, list) or not letters or not all(isinstance(letter, str) for letter in letters)
                or len(letters) != len(set(letters)) or not set(letters) <= mapping.keys()
                or (payload["question_type"] == "single_choice" and len(letters) != 1)):
            counts["invalid"] += 1
        else:
            correct = score_options(letters, mapping, payload["answer_option_ids"])
            counts["correct" if correct else "incorrect"] += 1
    return {"is_mock": mock, "total": len(questions), "counts": counts,
            "accuracy": counts["correct"] / len(questions)}
