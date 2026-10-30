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
    Calculate motion features for each tracked object.

    Speed: pixels/second.
    Acceleration: pixels/second squared.

    These are image-space estimates, not real-world measurements.
    """

    missing = REQUIRED_COLUMNS - set(detections.columns)

    if missing:
        raise ValueError(
            f"Tracking CSV is missing columns: {sorted(missing)}"
        )

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    df = detections.copy()

    for column in [
        "frame",
        "center_x",
        "center_y",
        "confidence",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "frame",
            "track_id",
            "class_name",
            "center_x",
            "center_y",
        ]
    )

    if df.empty:
        raise ValueError("No valid tracked detections found.")

    # A track may occasionally change its predicted class.
    # Keep its motion history together using track_id.
    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    motion_groups = []
    summary_rows = []

    for track_id, group in df.groupby(
        "track_id",
        sort=False,
    ):
        group = group.sort_values("frame").copy()

        # Use the most frequent detected class as the track label.
        class_counts = group["class_name"].value_counts()

        stable_class = (
            str(class_counts.index[0])
            if not class_counts.empty
            else "unknown"
        )

        frame_delta = group["frame"].diff()
        dt = frame_delta / fps

        dx = group["center_x"].diff()
        dy = group["center_y"].diff()

        displacement = np.hypot(dx, dy)

        speed = displacement / dt.where(dt > 0)

        direction = np.degrees(
            np.arctan2(dy, dx)
        )

        direction_change = (
            (direction.diff() + 180) % 360
        ) - 180

        # Approximate acceleration between consecutive speed estimates.
        acceleration = speed.diff() / dt.where(dt > 0)

        group["stable_class"] = stable_class
        group["delta_frames"] = frame_delta
        group["elapsed_seconds"] = (
            group["frame"] - group["frame"].min()
        ) / fps

        group["dx_pixels"] = dx
        group["dy_pixels"] = dy
        group["displacement_pixels"] = displacement
        group["speed_px_per_sec"] = speed
        group["direction_degrees"] = direction
        group["direction_change_degrees"] = direction_change
        group["acceleration_px_per_sec2"] = acceleration

        motion_groups.append(group)

        valid_speeds = speed.dropna()
        valid_accelerations = acceleration.dropna()

        first_frame = float(group["frame"].min())
        last_frame = float(group["frame"].max())

        span_frames = last_frame - first_frame + 1

        path_length = float(
            displacement.sum(skipna=True)
        )

        net_displacement = float(
            np.hypot(
                group["center_x"].iloc[-1]
                - group["center_x"].iloc[0],
                group["center_y"].iloc[-1]
                - group["center_y"].iloc[0],
            )
        )

        confidence_values = group["confidence"].dropna()

        summary_rows.append({
            "track_id": track_id,
            "class_name": stable_class,
            "first_frame": int(first_frame),
            "last_frame": int(last_frame),
            "observations": len(group),
            "track_span_seconds": (
                (last_frame - first_frame) / fps
            ),
            "track_coverage_ratio": (
                len(group) / span_frames
                if span_frames > 0
                else np.nan
            ),
            "path_length_pixels": path_length,
            "net_displacement_pixels": net_displacement,
            "mean_speed_px_per_sec": (
                float(valid_speeds.mean())
                if not valid_speeds.empty
                else np.nan
            ),
            "max_speed_px_per_sec": (
                float(valid_speeds.max())
                if not valid_speeds.empty
                else np.nan
            ),
            "mean_acceleration_px_per_sec2": (
                float(valid_accelerations.mean())
                if not valid_accelerations.empty
                else np.nan
            ),
            "max_abs_acceleration_px_per_sec2": (
                float(valid_accelerations.abs().max())
                if not valid_accelerations.empty
                else np.nan
            ),
            "mean_confidence": (
                float(confidence_values.mean())
                if not confidence_values.empty
                else np.nan
            ),
        })

    motion_df = pd.concat(
        motion_groups,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summary_rows)

    return motion_df, summary_df