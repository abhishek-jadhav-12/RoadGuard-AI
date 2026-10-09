
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
import time

from src.detection.detector import RoadObjectDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000001.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections"

CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640

# Road-related classes supported by the pretrained COCO model.
ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def main():
    print("=" * 60)
    print("RoadGuard AI - YOLO Road Object Detection")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Check the CCD video folder and filename."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInput video: {VIDEO_PATH.name}")
    print("Loading YOLO11n...")

    detector = RoadObjectDetector("yolo11n.pt")

    print("Model loaded.")
    print("Running detection; the first run may take longer.\n")

    class_counts = Counter()
    total_frames = 0
    total_detections = 0
    total_road_detections = 0
    inference_ms = []

    start_time = time.perf_counter()

    results = detector.predict(
        source=str(VIDEO_PATH),
        confidence=CONFIDENCE_THRESHOLD,
        image_size=IMAGE_SIZE,
        stream=True,
        save=True,
        project=str(OUTPUT_DIR),
        name="ccd_sample",
        exist_ok=True,
    )

    for result in results:
        total_frames += 1

        if result.boxes is None or len(result.boxes) == 0:
            continue

        total_detections += len(result.boxes)

        if result.speed and "inference" in result.speed:
            inference_ms.append(result.speed["inference"])

        for class_id in result.boxes.cls.tolist():
            class_name = result.names[int(class_id)]
            class_counts[class_name] += 1

            if class_name in ROAD_CLASSES:
                total_road_detections += 1

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Frames processed       : {total_frames}")
    print(f"All object detections  : {total_detections}")
    print(f"Road-object detections : {total_road_detections}")
    print(f"Elapsed time           : {elapsed:.2f} seconds")

    if total_frames:
        print(f"Processing throughput  : {total_frames / elapsed:.2f} FPS")

    if inference_ms:
        print(
            "Mean model inference  : "
            f"{sum(inference_ms) / len(inference_ms):.2f} ms/frame"
        )

    print("\nDetections by class (counts across frames):")
    if class_counts:
        for name, count in class_counts.most_common():
            print(f"  {name:<15}: {count}")
    else:
        print("  No objects detected above the confidence threshold.")

    print(f"\nAnnotated video folder: {OUTPUT_DIR / 'ccd_sample'}")
    print("\nDetection completed.")


if __name__ == "__main__":
    main()