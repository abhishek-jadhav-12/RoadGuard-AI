from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
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
    / "stable_tracking_results.csv"
)

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError("Tracking results are empty.")

if "track_id" not in df or "class_name" not in df:
    raise ValueError("CSV must contain track_id and class_name columns.")

# Keep the original predictions and add a corrected class column.
df["stable_class"] = df["class_name"]

# Determine the most frequently predicted class for each track.
for track_id, indices in df.groupby("track_id", dropna=False).groups.items():
    if pd.isna(track_id):
        continue

    track_classes = df.loc[indices, "class_name"]
    majority_class = track_classes.mode()

    if not majority_class.empty:
        df.loc[indices, "stable_class"] = majority_class.iloc[0]

df.to_csv(OUTPUT_CSV, index=False)

print("\nOriginal predictions:")
print(df["class_name"].value_counts())

print("\nStable classes:")
print(df["stable_class"].value_counts())

print(f"\nSaved results to: {OUTPUT_CSV}")