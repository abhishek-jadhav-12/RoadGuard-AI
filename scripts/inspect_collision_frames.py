
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    ROOT / "data" / "raw" / "CCD" / "videos"
    / "Crash-1500" / "000004.mp4"
)

OUTPUT_DIR = ROOT / "outputs" / "incidents" / "frame_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inspect a short window around the flagged frame.
TARGET_FRAMES = range(27, 34)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO_PATH.name}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")

for frame_number in TARGET_FRAMES:
    # OpenCV uses zero-based frame indices.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"Saved: {output_path}")

cap.release()
print("Frame inspection images saved.")
