
import sys
import time
import json
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.tracker import RoadObjectTracker


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000001.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections" / "tracking"
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO11n + ByteTrack...")
    tracker = RoadObjectTracker()

    rows = []
    class_counts = Counter()
    start = time.perf_counter()

    results = tracker.track_video(
        VIDEO_PATH,
        confidence=0.40,
        image_size=640,
        save=True,
        output_dir=str(OUTPUT_DIR),
    )

    for frame_number, result in enumerate(results, start=1):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(class_ids)
        )

        for box, class_id, confidence, track_id in zip(
            xyxy, class_ids, confidences, track_ids
        ):
            class_name = result.names[int(class_id)]

            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            rows.append({
                "frame": frame_number,
                "track_id": track_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
            })
            class_counts[class_name] += 1

    elapsed = time.perf_counter() - start

    columns = [
        "frame", "track_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y"
    ]
    df = pd.DataFrame(rows, columns=columns)

    csv_path = OUTPUT_DIR / "tracking_results.csv"
    df.to_csv(csv_path, index=False)

    unique_ids = (
        df.loc[df["track_id"].notna(), "track_id"].nunique()
        if not df.empty else 0
    )

    summary = {
        "video": VIDEO_PATH.name,
        "frames_with_recorded_detections": (
            int(df["frame"].nunique()) if not df.empty else 0
        ),
        "road_object_detections": len(df),
        "unique_track_ids": int(unique_ids),
        "elapsed_seconds": round(elapsed, 3),
        "class_detection_counts": dict(class_counts),
        "note": (
            "Track IDs can be missed or switched. "
            "Counts are not ground-truth tracking accuracy."
        ),
    }

    summary_path = OUTPUT_DIR / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\n========== TRACKING SUMMARY ==========")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Recorded road-object detections: {len(df)}")
    print(f"Unique track IDs: {unique_ids}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print("\nDetections by class:")
    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")

    print(f"\nTracking CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print("Tracking completed.")


if __name__ == "__main__":
    main()