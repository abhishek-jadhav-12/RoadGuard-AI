
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000004.mp4"
)

OUTPUT_PATH = PROJECT_ROOT / "frame28_yolo_predictions.jpg"


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    # Load the existing pretrained model.
    model = YOLO("yolo11n.pt")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 27)

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError("Could not read frame 28.")

    # Run detection without ByteTrack.
    results = model.predict(
        source=frame,
        conf=0.10,
        imgsz=640,
        verbose=False,
    )

    result = results[0]

    print("\nYOLO raw predictions for frame 28")
    print("-" * 50)

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
    else:
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"Detection {index + 1}: "
                f"class={class_name}, "
                f"confidence={confidence:.4f}, "
                f"box=({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )

    # Save an annotated image for inspection.
    annotated = result.plot()
    cv2.imwrite(str(OUTPUT_PATH), annotated)

    print(f"\nAnnotated image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
