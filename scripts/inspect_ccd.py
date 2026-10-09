from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CCD_ROOT = PROJECT_ROOT / "data" / "raw" / "CCD"
VIDEO_ROOT = CCD_ROOT / "videos"

CRASH_DIR = VIDEO_ROOT / "Crash-1500"
NORMAL_DIR = VIDEO_ROOT / "Normal"


def count_videos(directory: Path) -> int:
    if not directory.exists():
        return 0

    return len(list(directory.glob("*.mp4")))


def main():
    print("=" * 60)
    print("RoadGuard AI - CCD Dataset Inspection")
    print("=" * 60)

    print(f"\nCCD directory:")
    print(CCD_ROOT)

    print(f"\nCCD exists: {CCD_ROOT.exists()}")
    print(f"Videos directory exists: {VIDEO_ROOT.exists()}")

    crash_count = count_videos(CRASH_DIR)
    normal_count = count_videos(NORMAL_DIR)

    print("\nDataset statistics")
    print("-" * 60)
    print(f"Crash videos : {crash_count}")
    print(f"Normal videos: {normal_count}")
    print(f"Total videos : {crash_count + normal_count}")

    annotation_file = CCD_ROOT / "Crash-1500.txt"

    print(f"\nAnnotation file:")
    print(annotation_file)
    print(f"Exists: {annotation_file.exists()}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()