from pathlib import Path
import cv2

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "incidents"
    / "frame_inspection_000004"
)

TARGET_FRAMES = [27, 28, 29, 30, 31, 32, 33]

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Total frames: {total_frames}")

    for frame_number in TARGET_FRAMES:
        if frame_number < 1 or frame_number > total_frames:
            print(f"Skipping invalid frame: {frame_number}")
            continue

        # OpenCV uses zero-based frame positions.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

    cap.release()
    print(f"\nFinished. Open this folder:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
