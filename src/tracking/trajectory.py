
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame",
    "track_id",
    "class_name",
    "confidence",
    "center_x",
    "center_y",
}


def calculate_trajectory_features(
    detections: pd.DataFrame,
    fps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate per-observation motion features and per-track summaries.

    Speed is measured in pixels/second, and acceleration in pixels/second².
    These are image-space estimates, not calibrated physical measurements.
    """
    missing = REQUIRED_COLUMNS - set(detections.columns)
    if missing:
        raise ValueError(f"Tracking CSV is missing columns: {sorted(missing)}")

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    df = detections.copy()

    for column in ["frame", "center_x", "center_y", "confidence"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["frame", "center_x", "center_y", "track_id", "class_name"]
    )

    if df.empty:
        raise ValueError("No valid tracked detections found.")

    # Track IDs can be reused across videos. This script analyzes one video.
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    output_groups = []
    summary_rows = []

    for (track_id, class_name), group in df.groupby(
        ["track_id", "class_name"], sort=False
    ):
        group = group.sort_values("frame").copy()

        frame_delta = group["frame"].diff()
        dt = frame_delta / fps

        dx = group["center_x"].diff()
        dy = group["center_y"].diff()

        displacement = np.hypot(dx, dy)
        speed = displacement / dt.where(dt > 0)

        # Image coordinates have y increasing downward.
        direction = np.degrees(np.arctan2(dy, dx))

        # Wrapped angular change, between -180 and +180 degrees.
        direction_change = (
            (direction.diff() + 180) % 360
        ) - 180

        # Acceleration proxy from changes in measured image-space speed.
        acceleration = speed.diff() / dt.where(dt > 0)

        group["delta_frames"] = frame_delta
        group["elapsed_seconds"] = group["frame"] / fps
        group["dx_pixels"] = dx
        group["dy_pixels"] = dy
        group["displacement_pixels"] = displacement
        group["speed_px_per_sec"] = speed
        group["direction_degrees"] = direction
        group["direction_change_degrees"] = direction_change
        group["acceleration_px_per_sec2"] = acceleration

        output_groups.append(group)

        valid_speeds = speed.dropna()
        valid_accelerations = acceleration.dropna()
        span_frames = (
            float(group["frame"].max() - group["frame"].min()) + 1
        )
        path_length = float(displacement.sum(skipna=True))
        net_displacement = float(
            np.hypot(
                group["center_x"].iloc[-1] - group["center_x"].iloc[0],
                group["center_y"].iloc[-1] - group["center_y"].iloc[0],
            )
        )

        summary_rows.append({
            "track_id": track_id,
            "class_name": class_name,
            "first_frame": int(group["frame"].min()),
            "last_frame": int(group["frame"].max()),
            "observations": len(group),
            "track_span_seconds": (
                (group["frame"].max() - group["frame"].min()) / fps
            ),
            "track_coverage_ratio": (
                len(group) / span_frames if span_frames > 0 else np.nan
            ),
            "path_length_pixels": path_length,
            "net_displacement_pixels": net_displacement,
            "mean_speed_px_per_sec": (
                float(valid_speeds.mean()) if not valid_speeds.empty else np.nan
            ),
            "max_speed_px_per_sec": (
                float(valid_speeds.max()) if not valid_speeds.empty else np.nan
            ),
            "mean_acceleration_px_per_sec2": (
                float(valid_accelerations.mean())
                if not valid_accelerations.empty else np.nan
            ),
            "max_abs_acceleration_px_per_sec2": (
                float(valid_accelerations.abs().max())
                if not valid_accelerations.empty else np.nan
            ),
            "mean_confidence": float(group["confidence"].mean()),
        })

    motion_df = pd.concat(output_groups, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return motion_df, summary_df
