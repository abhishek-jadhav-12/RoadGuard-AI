
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.accident.collision import analyze_collision_candidates


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000004.mp4"
)

TRACKING_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "tracking"
    / "tracking_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
    / "collision_analysis"
)


def read_video_info(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Could not determine video FPS or dimensions.")

    return fps, width, height


def main():
    print("=" * 60)
    print("RoadGuard AI - Collision Candidate Analysis")
    print("=" * 60)

    if not TRACKING_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACKING_CSV}\n"
            "Run python scripts\\run_tracking.py first."
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fps, width, height = read_video_info(VIDEO_PATH)
    detections = pd.read_csv(TRACKING_CSV)

    print(f"Video: {VIDEO_PATH.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Tracking records: {len(detections)}")

    pairs_df = analyze_collision_candidates(
    detections=detections,
    fps=fps,
    frame_width=width,
    frame_height=height,
    proximity_threshold=0.18,
    overlap_threshold=0.10,
    min_closing_speed_px_s=5.0,
    )

    all_pairs_path = OUTPUT_DIR / "collision_pair_analysis.csv"
    candidates_path = OUTPUT_DIR / "potential_collision_candidates.csv"
    summary_path = OUTPUT_DIR / "collision_analysis_summary.txt"

    pairs_df.to_csv(all_pairs_path, index=False)

    if pairs_df.empty:
        candidates_df = pairs_df.copy()
    else:
        candidates_df = pairs_df[
            pairs_df["potential_collision_candidate"]
        ].copy()

    candidates_df.to_csv(candidates_path, index=False)

    lines = [
        "RoadGuard AI - Collision Analysis Summary",
        "=" * 45,
        f"Video: {VIDEO_PATH.name}",
        f"Frames represented in pair analysis: "
        f"{pairs_df['frame'].nunique() if not pairs_df.empty else 0}",
        f"Object-pair observations: {len(pairs_df)}",
        f"Potential collision candidates: {len(candidates_df)}",
        "",
        "Important: candidates are heuristic flags for review,",
        "not confirmed collisions, accident probabilities, or",
        "validated emergency alerts.",
        "",
        f"All pair analysis: {all_pairs_path}",
        f"Candidate records: {candidates_path}",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== COLLISION ANALYSIS ==========")
    print(f"Object-pair observations: {len(pairs_df)}")
    print(f"Potential candidate rows: {len(candidates_df)}")

    if not candidates_df.empty:
        print("\nCandidate preview:")
        print(
            candidates_df[
                [
                    "frame",
                    "track_id_a",
                    "class_a",
                    "track_id_b",
                    "class_b",
                    "bbox_iou",
                    "closing_speed_px_per_sec",
                    "candidate_score",
                ]
            ].head(10).to_string(index=False)
        )
    else:
        print(
            "\nNo candidate rows were flagged. This does not prove "
            "that the video contains no collision."
        )

    print("\nSaved files:")
    print(all_pairs_path)
    print(candidates_path)
    print(summary_path)
    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
