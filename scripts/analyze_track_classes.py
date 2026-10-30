import json
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
    / "tracking_results_with_stable_classes.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "tracking"
    / "track_class_summary.json"
)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {"track_id", "class_name", "confidence", "frame"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["stable_class"] = ""
    df["class_consistency"] = ""

    summaries = []

    for track_id, group in df.groupby("track_id", sort=True):
        valid = group.dropna(subset=["class_name", "confidence"])

        if valid.empty:
            continue

        # Use confidence-weighted votes to estimate the dominant class.
        votes = (
            valid.assign(
                weighted_confidence=valid["confidence"].clip(0, 1)
            )
            .groupby("class_name")["weighted_confidence"]
            .sum()
            .sort_values(ascending=False)
        )

        stable_class = str(votes.index[0])

        # Keep the original labels; report consistency separately.
        class_counts = valid["class_name"].value_counts()
        total = int(class_counts.sum())
        dominant_count = int(class_counts.iloc[0])
        consistency = dominant_count / total if total else 0.0

        df.loc[group.index, "stable_class"] = stable_class
        df.loc[group.index, "class_consistency"] = round(
            consistency, 4
        )

        summaries.append({
            "track_id": int(track_id),
            "detection_count": total,
            "first_frame": int(valid["frame"].min()),
            "last_frame": int(valid["frame"].max()),
            "stable_class": stable_class,
            "class_counts": {
                str(k): int(v)
                for k, v in class_counts.items()
            },
            "class_consistency": round(consistency, 4),
            "confidence_weighted_votes": {
                str(k): round(float(v), 4)
                for k, v in votes.items()
            },
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    OUTPUT_JSON.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    print("Track class analysis completed.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_JSON}")
    print("\nTrack summary:")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
    