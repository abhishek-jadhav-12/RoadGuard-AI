
from pathlib import Path
import sys
import json
import math
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.accident.severity import estimate_severity


INPUT_FILE = (
    ROOT
    / "outputs"
    / "detections"
    / "collision_analysis"
    / "potential_collision_candidates.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "incidents"
OUTPUT_FILE = OUTPUT_DIR / "incident_report.json"


def safe_float(value):
    """Convert a CSV value to float, treating missing values as None."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Collision candidate file not found: {INPUT_FILE}\n"
            "Run scripts\\analyze_collisions.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    incidents = []

    for _, row in df.iterrows():
        if not bool(row.get("potential_collision_candidate", False)):
            continue

        bbox_iou = safe_float(row.get("bbox_iou")) or 0.0
        closing_speed = safe_float(
            row.get("closing_speed_px_per_sec")
        )

        # These remain conservative until validated from motion data.
        assessment = estimate_severity(
            bbox_iou=bbox_iou,
            closing_speed_px_s=closing_speed,
            sudden_motion_change=False,
            involved_objects=2,
            track_reliable=False,
        )

        incidents.append({
            "incident_id": f"INC-FRAME-{int(row['frame']):06d}",
            "source_video": "000004.mp4",
            "frame": int(row["frame"]),
            "objects": [
                {
                    "track_id": int(row["track_id_a"]),
                    "class": str(row["class_a"]),
                },
                {
                    "track_id": int(row["track_id_b"]),
                    "class": str(row["class_b"]),
                },
            ],
            "collision_evidence": {
                "bbox_iou": bbox_iou,
                "closing_speed_px_per_sec": closing_speed,
                "candidate_score": safe_float(
                    row.get("candidate_score")
                ),
                "candidate_score_is_probability": False,
            },
            "visual_risk_assessment": assessment.to_dict(),
            "confirmed_accident": False,
            "human_review_required": True,
        })

    report = {
        "report_generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "RoadGuard-AI",
        "report_type": "Preliminary visual incident analysis",
        "candidate_count": len(incidents),
        "incidents": incidents,
        "limitations": [
            "Candidates are not confirmed accidents.",
            "Visual risk scores are heuristic, not probabilities.",
            "Image-space motion is not real-world speed.",
            "No injuries or vehicle damage are inferred.",
            "Human review is required before taking action.",
        ],
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Incident report saved: {OUTPUT_FILE}")
    print(f"Candidate incidents: {len(incidents)}")

    for incident in incidents:
        assessment = incident["visual_risk_assessment"]
        print(
            f"Frame {incident['frame']}: "
            f"{assessment['level']} "
            f"(score={assessment['score']}, "
            f"confidence={assessment['confidence']})"
        )


if __name__ == "__main__":
    main()
