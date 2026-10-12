
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.trajectory import calculate_trajectory_features


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CCD"
    / "videos"
    / "Crash-1500"
    / "000001.mp4"
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
    / "tracking"
    / "trajectory_analysis"
)


def get_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if fps <= 0:
        raise ValueError(f"Could not determine FPS for {video_path}")

    return float(fps)


def save_trajectory_plot(
    motion_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_tracks = 0

    for (track_id, class_name), group in motion_df.groupby(
        ["track_id", "class_name"]
    ):
        group = group.sort_values("frame")

        if len(group) < 2:
            continue

        ax.plot(
            group["center_x"],
            group["center_y"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{class_name} ID {track_id}",
        )

        ax.annotate(
            "Start",
            (group["center_x"].iloc[0], group["center_y"].iloc[0]),
            fontsize=8,
        )
        plotted_tracks += 1

    ax.set_title("RoadGuard AI - Object Trajectories")
    ax.set_xlabel("Image X coordinate (pixels)")
    ax.set_ylabel("Image Y coordinate (pixels)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    if 0 < plotted_tracks <= 15:
        ax.legend(fontsize=8, loc="best")

    if plotted_tracks == 0:
        ax.text(
            0.5,
            0.5,
            "No tracks with two or more observations",
            transform=ax.transAxes,
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    print("=" * 60)
    print("RoadGuard AI - Trajectory and Motion Analysis")
    print("=" * 60)

    if not TRACKING_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACKING_CSV}\n"
            "Run python scripts\\run_tracking.py first."
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fps = get_video_fps(VIDEO_PATH)
    detections = pd.read_csv(TRACKING_CSV)

    print(f"\nInput video: {VIDEO_PATH.name}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Tracking records loaded: {len(detections)}")

    motion_df, summary_df = calculate_trajectory_features(
        detections,
        fps=fps,
    )

    motion_path = OUTPUT_DIR / "per_detection_motion.csv"
    summary_path = OUTPUT_DIR / "track_motion_summary.csv"
    plot_path = OUTPUT_DIR / "trajectory_plot.png"

    motion_df.to_csv(motion_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    save_trajectory_plot(motion_df, plot_path)

    print("\n========== MOTION SUMMARY ==========")
    print(f"Valid detection records: {len(motion_df)}")
    print(f"Tracks summarized: {len(summary_df)}")

    if not summary_df.empty:
        print(
            "Tracks with speed estimates: "
            f"{summary_df['mean_speed_px_per_sec'].notna().sum()}"
        )

        print("\nTop tracks by path length:")
        print(
            summary_df.sort_values(
                "path_length_pixels",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )

    print("\nSaved outputs:")
    print(f"Motion records: {motion_path}")
    print(f"Track summaries: {summary_path}")
    print(f"Trajectory plot: {plot_path}")
    print("\nTrajectory analysis completed.")


if __name__ == "__main__":
    main()
