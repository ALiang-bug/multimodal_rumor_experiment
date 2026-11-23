"""Derive gold status from finalized, version-bound human adjudications."""


def project_gold(records, required_votes):
    targets = {row[row["record_type"] + "_id"]: row for row in records
               if row["record_type"] in {"annotation", "qa"}}
    votes = {row["human_vote_id"]: row for row in records if row["record_type"] == "human_vote"}
    adjudications = [row for row in records if row["record_type"] == "adjudication"]
    for kind in ("annotation", "qa"):
        for target in targets.values():
            if target["record_type"] != kind or target["status"] in {"STALE", "EXCLUDED", "SKIPPED"}:
                continue
            if kind == "qa":
                upstream = targets[target["source_annotation_id"]]
                if (upstream["status"] != "ACCEPTED_GOLD"
                        or target["source_annotation_content_hash"] != upstream["content_hash"]):
                    continue
            matching = []
            for row in adjudications:
                if row["outcome"] == "CORRECTED":
                    binding = targets.get(target["supersedes_id"])
                    if target["adjudication_id"] != row["adjudication_id"]:
                        continue
                else:
                    binding = target if row["outcome"] == "CONFIRMED" else None
                if (not binding or row["target_record_type"] != kind or row["target_id"] != binding[kind + "_id"]
                        or row["target_content_hash"] != binding["content_hash"]
                        or row["status"] != "FINALIZED"
                        or not row["eligible_for_gold"] or row["is_mock"] != target["is_mock"]
                        or row["final_" + kind + "_payload"] != target["payload"]):
                    continue
                eligible = []
                for vote_id in row["human_vote_ids"]:
                    vote = votes.get(vote_id)
                    if (vote and vote["target_record_type"] == kind and vote["target_id"] == row["target_id"]
                            and vote["target_content_hash"] == row["target_content_hash"]
                            and vote["is_mock"] == row["is_mock"] and vote["status"] == "SUBMITTED"
                            and vote["vote_mode"] == "independent" and vote["blinded_to_model"]
                            and vote["blinded_to_other_humans"]):
                        eligible.append(vote)
                independent = {vote["annotator_id"] for vote in eligible}
                if len(independent) >= required_votes:
                    matching.append((row, eligible))
            if len(matching) == 1:
                row, eligible = matching[0]
                target.update(
                    status="ACCEPTED_GOLD", quality_tier="gold",
                    human_vote_ids=[vote["human_vote_id"] for vote in eligible],
                    adjudication_id=row["adjudication_id"],
                )
