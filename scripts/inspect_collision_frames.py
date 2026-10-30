from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACKING_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "tracking"
    / "tracking_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "tracking"
    / "trajectory_analysis"
    / "collision_frame_details.csv"
)


def main():
    if not TRACKING_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACKING_CSV}"
        )

    df = pd.read_csv(TRACKING_CSV)

    required_columns = {
        "frame",
        "track_id",
        "class_name",
        "confidence",
        "center_x",
        "center_y",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    df["track_id"] = pd.to_numeric(
        df["track_id"],
        errors="coerce",
    )

    # Inspect a small window around the suspicious frame.
    window = df[
        df["frame"].between(28, 32)
    ].copy()

    if window.empty:
        print("No detections found in frames 28-32.")
        return

    columns = [
        "frame",
        "track_id",
        "class_name",
        "confidence",
        "center_x",
        "center_y",
    ]

    # Include bounding boxes when available.
    for column in [
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]:
        if column in window.columns:
            columns.append(column)

    window = window[columns].sort_values(
        ["frame", "track_id"]
    )

    print("\n========== FRAMES 28-32 ==========")
    print(window.to_string(index=False))

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved diagnostic CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()