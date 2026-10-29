import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.review_db import get_review, save_review


def main():
    incident_id = "INC-FRAME-000030"

    print("\nBefore saving:")
    print(get_review(incident_id))

    saved_review = save_review(
        incident_id=incident_id,
        review_status="Confirmed",
        reviewer_notes="Testing the incident review database."
    )

    print("\nAfter saving:")
    print(saved_review)

    retrieved_review = get_review(incident_id)

    assert retrieved_review["review_status"] == "Confirmed"
    assert retrieved_review["reviewer_notes"] == (
        "Testing the incident review database."
    )

    print("\nSUCCESS: Database save and retrieval are working!")


if __name__ == "__main__":
    main()