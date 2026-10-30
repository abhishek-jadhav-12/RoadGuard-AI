from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

csv_path = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "tracking"
    / "tracking_results.csv"
)

if not csv_path.exists():
    raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

print("=" * 60)
print("ROADGUARD AI - TRACKING INSPECTION")
print("=" * 60)

print(f"\nTotal detections: {len(df)}")
print(f"Unique frames: {df['frame'].nunique()}")
print(f"Unique tracks: {df['track_id'].nunique()}")

print("\nTRACK SUMMARY")
print("-" * 60)

summary = (
    df.groupby("track_id")
    .agg(
        detections=("frame", "count"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        classes=(
            "class_name",
            lambda values: ", ".join(
                sorted(set(values.dropna().astype(str)))
            ),
        ),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\nFRAME-BY-FRAME DETECTIONS")
print("-" * 60)

columns = [
    "frame",
    "track_id",
    "class_name",
    "center_x",
    "center_y",
]

print(
    df[columns]
    .sort_values(["frame", "track_id"])
    .to_string(index=False)
)

print("\nInspection completed.")