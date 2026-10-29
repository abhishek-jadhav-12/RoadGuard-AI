
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.accident.severity import estimate_severity


cases = [
    {
        "name": "Missing motion evidence",
        "bbox_iou": 0.976745,
        "closing_speed_px_s": None,
        "track_reliable": False,
    },
    {
        "name": "Moderate overlap and motion evidence",
        "bbox_iou": 0.30,
        "closing_speed_px_s": 20.0,
        "track_reliable": True,
    },
    {
        "name": "No collision cues",
        "bbox_iou": 0.0,
        "closing_speed_px_s": None,
        "track_reliable": False,
    },
]

for case in cases:
    name = case.pop("name")
    result = estimate_severity(**case)

    print(f"\n=== {name} ===")
    print(json.dumps(result.to_dict(), indent=2))
