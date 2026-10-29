import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "outputs" / "detections" / "tracking" / "tracked_video"
OUTPUT_DIR = INPUT_DIR / "converted_mp4"

VIDEO_EXTENSIONS = {".avi", ".mov", ".mkv", ".mp4"}


def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = [
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"No source videos found in: {INPUT_DIR}")
        return

    converted = 0
    failed = 0

    for source in videos:
        destination = OUTPUT_DIR / f"{source.stem}.mp4"

        print(f"\nConverting: {source.name}")

        command = [
            ffmpeg_exe,
            "-y",
            "-i", str(source),
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(destination),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and destination.exists():
                if destination.stat().st_size > 0:
                    print(f"SUCCESS: {destination}")
                    converted += 1
                    continue

            print(f"FAILED: {source.name}")
            print(result.stderr[-2500:])
            failed += 1

            if destination.exists() and destination.stat().st_size == 0:
                destination.unlink()

        except OSError as exc:
            print(f"ERROR converting {source.name}: {exc}")
            failed += 1

    print("\n========== CONVERSION SUMMARY ==========")
    print(f"Successfully converted: {converted}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()