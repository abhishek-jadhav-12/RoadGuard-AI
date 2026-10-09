
import sys
import json
import time
from pathlib import Path
from collections import Counter

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
    / "000001.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections" / "evaluation"
CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO11n...")
    detector = RoadObjectDetector("yolo11n.pt")

    class_counts = Counter()
    frame_records = []
    inference_times = []
    all_confidences = []
    total_detections = 0

    start = time.perf_counter()

    results = detector.predict(
        source=str(VIDEO_PATH),
        confidence=CONFIDENCE_THRESHOLD,
        image_size=IMAGE_SIZE,
        stream=True,
        save=True,
        project=str(OUTPUT_DIR),
        name="annotated_video",
        exist_ok=True,
    )

    for frame_number, result in enumerate(results, start=1):
        boxes = result.boxes
        frame_count = 0
        frame_confidences = []

        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()

            for class_id, confidence in zip(class_ids, confidences):
                class_name = result.names[int(class_id)]
                class_counts[class_name] += 1
                frame_confidences.append(float(confidence))
                all_confidences.append(float(confidence))

            frame_count = len(class_ids)
            total_detections += frame_count

        inference_ms = None
        if result.speed and "inference" in result.speed:
            inference_ms = float(result.speed["inference"])
            inference_times.append(inference_ms)

        frame_records.append({
            "frame_number": frame_number,
            "detections": frame_count,
            "mean_confidence": (
                sum(frame_confidences) / len(frame_confidences)
                if frame_confidences else None
            ),
            "inference_ms": inference_ms,
        })

    elapsed = time.perf_counter() - start
    frame_df = pd.DataFrame(frame_records)
    csv_path = OUTPUT_DIR / "per_frame_detections.csv"
    frame_df.to_csv(csv_path, index=False)

    summary = {
        "video": VIDEO_PATH.name,
        "frames_processed": len(frame_records),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "total_detections_across_frames": total_detections,
        "mean_confidence": (
            sum(all_confidences) / len(all_confidences)
            if all_confidences else None
        ),
        "mean_inference_ms": (
            sum(inference_times) / len(inference_times)
            if inference_times else None
        ),
        "elapsed_seconds": round(elapsed, 3),
        "end_to_end_fps": (
            round(len(frame_records) / elapsed, 2)
            if elapsed > 0 else None
        ),
        "detections_by_class": dict(class_counts),
        "note": (
            "These are inference statistics, not accuracy or mAP. "
            "Ground-truth bounding-box annotations are needed to "
            "calculate detection precision, recall, and mAP."
        ),
    }

    json_path = OUTPUT_DIR / "detection_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== YOLO EVALUATION ==========")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Frames processed: {len(frame_records)}")
    print(f"Total detections across frames: {total_detections}")
    print(f"Mean confidence: {summary['mean_confidence']}")
    print(f"Mean inference time (ms): {summary['mean_inference_ms']}")
    print(f"End-to-end FPS: {summary['end_to_end_fps']}")

    print("\nDetections by class:")
    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count}")

    print(f"\nFrame-level CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()