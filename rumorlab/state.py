"""SQLite work leases and atomic result commits."""

import json
import sqlite3
import time
import uuid

from .validation import canonical_hash


class State:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS imports (
                run_id TEXT PRIMARY KEY, digest TEXT NOT NULL,
                media_root TEXT NOT NULL, records TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS import_manifests (
                run_id TEXT PRIMARY KEY REFERENCES imports(run_id), manifest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS split_plans (
                run_id TEXT PRIMARY KEY, digest TEXT NOT NULL, plan TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mock_exports (
                run_id TEXT PRIMARY KEY, records TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS human_submissions (
                run_id TEXT NOT NULL, record_id TEXT NOT NULL, record TEXT NOT NULL,
                PRIMARY KEY (run_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY, job_id TEXT NOT NULL,
                state TEXT NOT NULL, timestamp REAL NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, stage TEXT NOT NULL, subject TEXT NOT NULL,
                task TEXT NOT NULL, input_hash TEXT NOT NULL,
                state TEXT NOT NULL, lease TEXT, expires REAL,
                result TEXT, technical_retries INTEGER NOT NULL DEFAULT 0,
                format_repairs INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS dependencies (
                job_id TEXT NOT NULL, record_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (job_id, record_id, content_hash)
            );
            CREATE INDEX IF NOT EXISTS dependencies_by_record ON dependencies(record_id, content_hash);
            CREATE TABLE IF NOT EXISTS invalidated_versions (
                record_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                PRIMARY KEY (record_id, content_hash)
            );
        """)

    def close(self):
        self.db.close()

    def _event(self, job_id, state, detail):
        self.db.execute(
            "INSERT INTO events(job_id, state, timestamp, detail) VALUES (?, ?, ?, ?)",
            (job_id, state, time.time(), json.dumps(detail, ensure_ascii=False)),
        )

    def claim(self, stage, subject, task, input_hash, dependencies, lease_seconds):
        dependencies = sorted(set(dependencies))
        job_id = canonical_hash([stage, subject, task, input_hash, dependencies])
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for record_id, digest in dependencies:
                retired = self.db.execute(
                    "SELECT 1 FROM invalidated_versions WHERE record_id = ? AND content_hash = ?",
                    (record_id, digest),
                ).fetchone()
                if retired:
                    raise ValueError(f"Stale dependency: {record_id} at {digest}")
            created = self.db.execute(
                """INSERT OR IGNORE INTO jobs(id, stage, subject, task, input_hash, state)
                   VALUES (?, ?, ?, ?, ?, 'QUEUED')""",
                (job_id, stage, subject, task, input_hash),
            )
            if created.rowcount:
                self._event(job_id, "QUEUED", {})
                self.db.executemany(
                    "INSERT INTO dependencies VALUES (?, ?, ?)",
                    [(job_id, record_id, digest) for record_id, digest in dependencies],
                )
            job = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job["state"] == "SUCCEEDED":
                return job_id, None, json.loads(job["result"])
            now = time.time()
            if job["state"] == "RUNNING":
                if job["expires"] > now:
                    raise ValueError(f"Job already leased: {job_id}")
                if job["technical_retries"] >= 2:
                    self.db.execute("UPDATE jobs SET state = 'FAILED', lease = NULL, expires = NULL WHERE id = ?", (job_id,))
                    self._event(job_id, "FAILED", {"category": "technical", "reason": "lease expired"})
                    return job_id, None, None
                self.db.execute("UPDATE jobs SET technical_retries = technical_retries + 1 WHERE id = ?", (job_id,))
                self._event(job_id, "RETRYABLE_ERROR", {"category": "technical", "reason": "lease expired"})
            elif job["state"] not in {"QUEUED", "RETRYABLE_ERROR"}:
                return job_id, None, None
            lease = uuid.uuid4().hex
            self.db.execute(
                "UPDATE jobs SET state = 'RUNNING', lease = ?, expires = ? WHERE id = ?",
                (lease, now + lease_seconds, job_id),
            )
            self._event(job_id, "RUNNING", {"lease": lease, "expires": now + lease_seconds})
        return job_id, lease, None

    def invalidate(self, record_id, content_hash):
        """Retire a specific version and revoke leases on all downstream jobs."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            return self._invalidate_in_transaction(record_id, content_hash)

    def _invalidate_in_transaction(self, record_id, content_hash):
        pending = [(record_id, content_hash)]
        stale = []
        while pending:
            version = pending.pop()
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO invalidated_versions VALUES (?, ?)", version,
            ).rowcount
            if not inserted:
                continue
            jobs = self.db.execute(
                """SELECT jobs.id, jobs.result FROM dependencies
                   JOIN jobs ON jobs.id = dependencies.job_id
                   WHERE record_id = ? AND content_hash = ? AND jobs.state != 'STALE'
                   ORDER BY jobs.id""", version,
            ).fetchall()
            for job in jobs:
                job_id = job["id"]
                self.db.execute(
                    "UPDATE jobs SET state = 'STALE', lease = NULL, expires = NULL WHERE id = ?",
                    (job_id,),
                )
                self._event(job_id, "STALE", {"record_id": version[0], "content_hash": version[1]})
                stale.append(job_id)
                if job["result"] is not None:
                    result = json.loads(job["result"])
                    if "content_hash" in result:
                        pending.append((job_id, result["content_hash"]))
        return stale

    def heartbeat(self, job_id, lease, lease_seconds):
        with self.db:
            now = time.time()
            updated = self.db.execute(
                """UPDATE jobs SET expires = ? WHERE id = ? AND lease = ?
                   AND state = 'RUNNING' AND expires > ?""",
                (now + lease_seconds, job_id, lease, now),
            )
            if updated.rowcount != 1:
                raise ValueError(f"Job lease lost: {job_id}")
            self._event(job_id, "RUNNING", {"lease": lease, "expires": now + lease_seconds})

    def finish(self, job_id, lease, *, result=None, error=None, retry_limit=0):
        if (result is None) == (error is None):
            raise ValueError("Supply exactly one of result and error")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            job = self.db.execute(
                "SELECT * FROM jobs WHERE id = ? AND lease = ? AND state = 'RUNNING' AND expires > ?",
                (job_id, lease, time.time()),
            ).fetchone()
            if job is None:
                raise ValueError(f"Job lease lost: {job_id}")
            state = "SUCCEEDED" if error is None else "FAILED"
            if error and error["category"] in {"technical", "format"}:
                category = error["category"]
                field, maximum = ("technical_retries", 2) if category == "technical" else ("format_repairs", 1)
                if job[field] < min(retry_limit, maximum):
                    self.db.execute(f"UPDATE jobs SET {field} = {field} + 1 WHERE id = ?", (job_id,))
                    state = "RETRYABLE_ERROR"
            encoded = json.dumps(result, ensure_ascii=False) if result is not None else None
            self.db.execute(
                "UPDATE jobs SET state = ?, lease = NULL, expires = NULL, result = ? WHERE id = ?",
                (state, encoded, job_id),
            )
            self._event(job_id, state, {"error": error, "result": result})
        return state
