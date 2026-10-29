import sys
import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import RoadObjectDetector


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000004.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "raw_diagnostics"
)

CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO11n...")
    detector = RoadObjectDetector("yolo11n.pt")

    records = []
    start = time.perf_counter()

    results = detector.predict(
        source=str(VIDEO_PATH),
        confidence=CONFIDENCE_THRESHOLD,
        image_size=IMAGE_SIZE,
        stream=True,
        save=False,
    )

    for frame_number, result in enumerate(results, start=1):
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for box, class_id, confidence in zip(
            xyxy, class_ids, confidences
        ):
            x1, y1, x2, y2 = box

            records.append({
                "frame": frame_number,
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

    df = pd.DataFrame(
        records,
        columns=[
            "frame",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )

    csv_path = OUTPUT_DIR / "000004_raw_detections.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - start

    summary = {
        "video": VIDEO_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "detection_records": len(df),
        "frames_with_detections": (
            int(df["frame"].nunique()) if not df.empty else 0
        ),
        "detections_by_class": (
            df["class_name"].value_counts().to_dict()
            if not df.empty else {}
        ),
        "elapsed_seconds": round(elapsed, 3),
        "note": (
            "Raw detector records, not ground-truth labels "
            "or validated detection accuracy."
        ),
    }

    summary_path = OUTPUT_DIR / "000004_raw_detection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nRaw detection diagnostic completed.")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Detection records: {len(df)}")
    print(f"Frames with detections: {summary['frames_with_detections']}")
    print("\nDetections by class:")

    if not df.empty:
        print(df["class_name"].value_counts().to_string())

    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
