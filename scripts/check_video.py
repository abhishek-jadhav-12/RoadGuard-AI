from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRASH_DIR = PROJECT_ROOT / "data" / "raw" / "CCD" / "videos" / "Crash-1500"


def main():
    videos = sorted(CRASH_DIR.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(
            f"No videos found in {CRASH_DIR}"
        )

    video_path = videos[0]

    print(f"Testing video: {video_path.name}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("Could not open video.")

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = frame_count / fps if fps > 0 else 0

    print("\nVideo information")
    print("-" * 40)
    print(f"File       : {video_path.name}")
    print(f"Frames     : {frame_count}")
    print(f"FPS        : {fps:.2f}")
    print(f"Resolution : {width} x {height}")
    print(f"Duration   : {duration:.2f} seconds")

    capture.release()


if __name__ == "__main__":
    main()