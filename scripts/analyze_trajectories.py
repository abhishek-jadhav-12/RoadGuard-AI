import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.trajectory import calculate_trajectory_features


# ============================================================
# CONFIGURATION
# ============================================================

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
    / "tracking"
    / "trajectory_analysis"
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path: Path) -> tuple[float, int, int]:
    """Return video FPS, width, and height."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise ValueError(
            f"Could not determine FPS for {video_path}"
        )

    return float(fps), width, height


# ============================================================
# TRAJECTORY VISUALIZATION
# ============================================================

def save_trajectory_plot(
    motion_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot each track as a continuous trajectory."""

    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_tracks = 0

    # Group by track ID only. This prevents class changes from
    # splitting a track into separate trajectory lines.
    for track_id, group in motion_df.groupby(
        "track_id",
        sort=False,
    ):
        group = group.sort_values("frame").copy()

        if len(group) < 2:
            continue

        # Prefer the stable class produced by trajectory.py.
        if (
            "stable_class" in group.columns
            and group["stable_class"].notna().any()
        ):
            class_name = str(
                group["stable_class"].dropna().iloc[0]
            )
        else:
            classes = group["class_name"].dropna().mode()
            class_name = (
                str(classes.iloc[0])
                if not classes.empty
                else "unknown"
            )

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
            (
                group["center_x"].iloc[0],
                group["center_y"].iloc[0],
            ),
            fontsize=8,
        )

        plotted_tracks += 1

    ax.set_title(
        "RoadGuard AI - Object Trajectories"
    )
    ax.set_xlabel("Image X coordinate (pixels)")
    ax.set_ylabel("Image Y coordinate (pixels)")

    # Image coordinates increase downward.
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
            va="center",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ============================================================
# TRACKING DATA DIAGNOSTICS
# ============================================================

def print_tracking_diagnostics(
    detections: pd.DataFrame,
    fps: float,
) -> None:
    """Print track counts, observation gaps, and class changes."""

    print("\n========== TRACKING DIAGNOSTICS ==========")

    print(f"Total detections: {len(detections)}")

    if "frame" not in detections.columns:
        print("Cannot inspect frame gaps: frame column is missing.")
        return

    print(
        "Unique frames observed: "
        f"{detections['frame'].nunique()}"
    )

    if "track_id" not in detections.columns:
        print("Cannot inspect tracks: track_id column is missing.")
        return

    print(
        "Unique track IDs: "
        f"{detections['track_id'].nunique()}"
    )

    for track_id, group in detections.groupby(
        "track_id",
        sort=False,
    ):
        group = group.sort_values("frame")

        frames = pd.to_numeric(
            group["frame"],
            errors="coerce",
        ).dropna()

        if frames.empty:
            continue

        frame_gaps = frames.diff().dropna()
        missing_frame_gaps = int(
            (frame_gaps > 1).sum()
        )

        observed_classes = (
            group["class_name"].dropna().unique().tolist()
            if "class_name" in group.columns
            else []
        )

        print(
            f"Track {track_id}: "
            f"{len(group)} observations, "
            f"frames {int(frames.min())}-"
            f"{int(frames.max())}, "
            f"gaps between observations: "
            f"{missing_frame_gaps}, "
            f"classes: {observed_classes}"
        )

    print(
        "\nNote: Frame gaps indicate missing observations "
        "between recorded detections. They do not by "
        "themselves prove tracking failure."
    )

    print(
        f"Motion timing uses the configured video FPS: "
        f"{fps:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RoadGuard AI - Trajectory and Motion Analysis")
    print("=" * 60)

    # Validate input files.
    if not TRACKING_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACKING_CSV}\n"
            "Run python scripts\\run_tracking.py first."
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load video information and tracking detections.
    fps, width, height = get_video_info(VIDEO_PATH)

    detections = pd.read_csv(TRACKING_CSV)

    print(f"\nInput video: {VIDEO_PATH.name}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Video resolution: {width}x{height}")
    print(f"Tracking records loaded: {len(detections)}")

    # Validate basic tracking data.
    required_columns = {
        "frame",
        "track_id",
        "class_name",
        "confidence",
        "center_x",
        "center_y",
    }

    missing_columns = required_columns - set(
        detections.columns
    )

    if missing_columns:
        raise ValueError(
            "Tracking CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Warn that the script cannot independently prove the CSV
    # belongs to the selected video.
    print(
        "\nIMPORTANT: Confirm that tracking_results.csv "
        f"was generated from {VIDEO_PATH.name}."
    )

    print_tracking_diagnostics(
        detections,
        fps,
    )

    # Calculate motion features.
    motion_df, summary_df = calculate_trajectory_features(
        detections,
        fps=fps,
    )

    # Output paths.
    motion_path = (
        OUTPUT_DIR / "per_detection_motion.csv"
    )

    summary_path = (
        OUTPUT_DIR / "track_motion_summary.csv"
    )

    plot_path = (
        OUTPUT_DIR / "trajectory_plot.png"
    )

    # Save results.
    motion_df.to_csv(
        motion_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    save_trajectory_plot(
        motion_df,
        plot_path,
    )

    # Display summary.
    print("\n========== MOTION SUMMARY ==========")

    print(
        f"Valid detection records: {len(motion_df)}"
    )

    print(
        f"Tracks summarized: {len(summary_df)}"
    )

    if not summary_df.empty:
        tracks_with_speed = (
            summary_df["mean_speed_px_per_sec"].notna().sum()
        )

        print(
            f"Tracks with speed estimates: {tracks_with_speed}"
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

    print("\nTrajectory analysis completed successfully.")


if __name__ == "__main__":
    main()