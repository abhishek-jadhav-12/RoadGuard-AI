
from pathlib import Path
import ast
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CCD_ROOT = PROJECT_ROOT / "data" / "raw" / "CCD"
ANNOTATION_FILE = CCD_ROOT / "Crash-1500.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ccd_metadata.csv"


def parse_annotation_line(line):
    """
    Parse a CCD annotation line while preserving commas
    inside the bracketed frame-label list.
    """
    line = line.strip()
    if not line:
        return None

    match = re.match(
        r"^\s*([^,]+),\s*(\[[^\]]*\])\s*,\s*(.*)$",
        line
    )

    if not match:
        return None

    video_id, labels_text, metadata_text = match.groups()

    try:
        labels = ast.literal_eval(labels_text)
        if not isinstance(labels, list):
            return None
        labels = [int(value) for value in labels]
    except (ValueError, SyntaxError, TypeError):
        return None

    metadata = [value.strip() for value in metadata_text.split(",")]

    # Expected remaining fields:
    # start_frame, youtube_id, timing, weather, ego_involved
    if len(metadata) != 5:
        return None

    start_frame, youtube_id, timing, weather, ego_involved = metadata

    first_accident_frame = next(
        (i for i, label in enumerate(labels) if label == 1),
        None
    )

    return {
        "video_id": video_id.strip(),
        "frame_labels": str(labels),
        "start_frame": start_frame,
        "youtube_id": youtube_id,
        "timing": timing,
        "weather": weather,
        "ego_involved": ego_involved,
        "frame_count": len(labels),
        "first_accident_frame": first_accident_frame,
        "accident_present": int(1 in labels),
    }


def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    records = []
    malformed_lines = []

    with open(
        ANNOTATION_FILE, "r", encoding="utf-8-sig", errors="replace"
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = parse_annotation_line(line)

            if record is None:
                malformed_lines.append(line_number)
            else:
                records.append(record)

    if not records:
        raise ValueError(
            "No annotations parsed. Check the annotation file format."
        )

    df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("CCD metadata validation")
    print("=" * 60)
    print(f"Successfully parsed rows: {len(df)}")
    print(f"Malformed lines skipped: {len(malformed_lines)}")
    print(f"Videos with positive labels: {df['accident_present'].sum()}")
    print(f"Videos with frame labels: {(df['frame_count'] > 0).sum()}")
    print(f"Total parsed frame labels: {df['frame_count'].sum()}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFirst five records:")
    print(
        df[
            [
                "video_id",
                "frame_count",
                "first_accident_frame",
                "accident_present",
                "timing",
                "weather",
            ]
        ].head().to_string(index=False)
    )

    if malformed_lines:
        print("\nFirst malformed line numbers:", malformed_lines[:10])


if __name__ == "__main__":
    main()