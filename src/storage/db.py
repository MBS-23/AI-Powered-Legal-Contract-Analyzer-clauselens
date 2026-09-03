"""
db.py
-----
Lightweight SQLite persistence so the app has a *contract library* rather than
losing everything at the end of a session. Each analyzed contract is stored
with its clauses, so the library can be browsed and searched across documents.

Design:
  - Pure stdlib sqlite3 — nothing to install, works locally and on Streamlit
    Cloud (the DB file lives on the app's ephemeral disk; that's fine for a
    demo/portfolio tool, and swapping in Postgres later would only touch this
    module).
  - Two tables, contracts (1) --< clauses (N), foreign key with ON DELETE
    CASCADE so removing a contract removes its clauses.
  - Every function opens and closes its own connection. That keeps things
    thread-safe under Streamlit's re-run model without a shared global handle.

The default DB path is data/contracts.db (gitignored). Tests pass an explicit
path to an isolated temp file.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "contracts.db"


@dataclass
class StoredClause:
    section_number: str
    heading: str
    clause_type: str | None
    confidence: float
    method: str
    text: str


@contextmanager
def _connect(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the schema if it doesn't exist. Safe to call repeatedly."""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                filename     TEXT NOT NULL,
                upload_date  TEXT NOT NULL,
                num_clauses  INTEGER NOT NULL DEFAULT 0,
                risk_score   INTEGER,
                risk_level   TEXT,
                summary      TEXT,
                entities     TEXT,           -- JSON blob
                full_text    TEXT
            );

            CREATE TABLE IF NOT EXISTS clauses (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id    INTEGER NOT NULL,
                section_number TEXT,
                heading        TEXT,
                clause_type    TEXT,
                confidence     REAL,
                method         TEXT,
                text           TEXT NOT NULL,
                FOREIGN KEY (contract_id) REFERENCES contracts(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_clauses_contract ON clauses(contract_id);
            CREATE INDEX IF NOT EXISTS idx_clauses_type ON clauses(clause_type);
            """
        )


def save_contract(
    filename: str,
    clauses,
    *,
    risk_score: int | None = None,
    risk_level: str | None = None,
    summary: str | None = None,
    entities: dict | None = None,
    full_text: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist one analyzed contract and its clauses. Returns the contract id.

    `clauses` is a list of objects with the ClauseMatch attributes
    (section_number, heading, clause_type, confidence, method, text).
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO contracts
                (filename, upload_date, num_clauses, risk_score, risk_level, summary, entities, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                len(clauses),
                risk_score,
                risk_level,
                summary,
                json.dumps(entities) if entities is not None else None,
                full_text,
            ),
        )
        contract_id = int(cur.lastrowid)

        conn.executemany(
            """
            INSERT INTO clauses
                (contract_id, section_number, heading, clause_type, confidence, method, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    contract_id,
                    getattr(c, "section_number", "") or "",
                    getattr(c, "heading", "") or "",
                    getattr(c, "clause_type", None),
                    float(getattr(c, "confidence", 0.0) or 0.0),
                    getattr(c, "method", "keyword"),
                    getattr(c, "text", "") or "",
                )
                for c in clauses
            ],
        )
        return contract_id


def list_contracts(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all stored contracts (metadata only), newest first."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, upload_date, num_clauses, risk_score, risk_level, summary
            FROM contracts ORDER BY id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_contract(contract_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict | None:
    """Return a single contract with its clauses, or None if not found."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
        if row is None:
            return None
        contract = dict(row)
        if contract.get("entities"):
            try:
                contract["entities"] = json.loads(contract["entities"])
            except json.JSONDecodeError:
                contract["entities"] = None
        clause_rows = conn.execute(
            "SELECT * FROM clauses WHERE contract_id = ? ORDER BY id", (contract_id,)
        ).fetchall()
        contract["clauses"] = [dict(r) for r in clause_rows]
        return contract


def get_all_clauses(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return every stored clause joined with its contract filename.

    This is the corpus the semantic-search index is built from.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT cl.id, cl.contract_id, c.filename, cl.section_number, cl.heading,
                   cl.clause_type, cl.confidence, cl.method, cl.text
            FROM clauses cl JOIN contracts c ON c.id = cl.contract_id
            ORDER BY cl.id
            """
        ).fetchall()
        return [dict(r) for r in rows]


def delete_contract(contract_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Delete a contract and (via cascade) its clauses."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))


def count_contracts(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with _connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0])
