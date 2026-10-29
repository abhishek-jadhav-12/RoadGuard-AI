from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data" / "processed"
DB_PATH = DB_DIR / "incident_reviews.db"

VALID_STATUSES = {
    "Pending Review",
    "Confirmed",
    "False Positive",
}


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incident_reviews (
                incident_id TEXT PRIMARY KEY,
                review_status TEXT NOT NULL DEFAULT 'Pending Review',
                reviewer_notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_review(incident_id):
    initialize_database()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT incident_id, review_status,
                   reviewer_notes, updated_at
            FROM incident_reviews
            WHERE incident_id = ?
            """,
            (incident_id,),
        ).fetchone()

    if row is None:
        return {
            "incident_id": incident_id,
            "review_status": "Pending Review",
            "reviewer_notes": "",
            "updated_at": None,
        }

    return dict(row)


def save_review(incident_id, review_status, reviewer_notes=""):
    if review_status not in VALID_STATUSES:
        raise ValueError("Invalid incident review status.")

    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO incident_reviews (
                incident_id,
                review_status,
                reviewer_notes,
                updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(incident_id) DO UPDATE SET
                review_status = excluded.review_status,
                reviewer_notes = excluded.reviewer_notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                incident_id,
                review_status,
                reviewer_notes.strip(),
            ),
        )

    return get_review(incident_id)