
from itertools import combinations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
}


def calculate_iou(box_a, box_b) -> float:
    """Calculate intersection-over-union for two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def analyze_collision_candidates(
    detections: pd.DataFrame,
    fps: float,
    frame_width: int,
    frame_height: int,
    proximity_threshold: float = 0.06,
    overlap_threshold: float = 0.10,
    min_closing_speed_px_s: float = 5.0,
) -> pd.DataFrame:
    """
    Generate heuristic potential-collision candidates for one video.

    Distances and closing speeds are measured in image coordinates.
    A candidate is not a confirmed collision or accident.
    """
    missing = REQUIRED_COLUMNS - set(detections.columns)
    if missing:
        raise ValueError(
            f"Tracking CSV is missing columns: {sorted(missing)}"
        )

    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("FPS and frame dimensions must be positive.")

    df = detections.copy()

    numeric_columns = [
        "frame", "track_id", "x1", "y1", "x2", "y2",
        "center_x", "center_y",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns + ["class_name"])
    df = df.drop_duplicates(subset=["frame", "track_id"])

    # Use the frame diagonal to normalize image-space center distance.
    frame_diagonal = float(np.hypot(frame_width, frame_height))
    previous_pair_distance = {}
    records = []

    for frame_number, frame_df in df.groupby("frame", sort=True):
        objects = frame_df.to_dict("records")

        for obj_a, obj_b in combinations(objects, 2):
            id_a = obj_a["track_id"]
            id_b = obj_b["track_id"]

            if id_a == id_b:
                continue

            # Canonical ordering keeps the same pair key across frames.
            if str(id_a) <= str(id_b):
                first, second = obj_a, obj_b
            else:
                first, second = obj_b, obj_a

            track_a = first["track_id"]
            track_b = second["track_id"]
            pair_key = (track_a, track_b)

            dx = float(second["center_x"] - first["center_x"])
            dy = float(second["center_y"] - first["center_y"])
            center_distance = float(np.hypot(dx, dy))
            normalized_distance = center_distance / frame_diagonal

            box_a = [first["x1"], first["y1"], first["x2"], first["y2"]]
            box_b = [second["x1"], second["y1"], second["x2"], second["y2"]]
            iou = calculate_iou(box_a, box_b)

            previous = previous_pair_distance.get(pair_key)
            closing_speed = np.nan

            if previous is not None:
                previous_frame, previous_distance = previous
                elapsed_seconds = (frame_number - previous_frame) / fps

                if elapsed_seconds > 0:
                    # Positive means the centers are getting closer.
                    closing_speed = (
                        previous_distance - center_distance
                    ) / elapsed_seconds

            previous_pair_distance[pair_key] = (
                frame_number,
                center_distance,
            )

            is_close = normalized_distance <= proximity_threshold
            boxes_overlap = iou >= overlap_threshold
            closing = (
                pd.notna(closing_speed)
                and closing_speed >= min_closing_speed_px_s
            )

            # Transparent heuristic score, not a calibrated probability.
            proximity_score = max(
                0.0,
                1.0 - normalized_distance / proximity_threshold,
            ) if proximity_threshold > 0 else 0.0

            closing_score = (
                min(
                    max(closing_speed, 0.0)
                    / (2 * min_closing_speed_px_s),
                    1.0,
                )
                if pd.notna(closing_speed)
                and min_closing_speed_px_s > 0
                else 0.0
            )

            score = (
                0.45 * min(iou, 1.0)
                + 0.35 * proximity_score
                + 0.20 * closing_score
            )

           
            # Flag strong overlap for review even when motion evidence is unavailable.
            strong_overlap = iou >= 0.70

            candidate = (
                strong_overlap
                or (boxes_overlap and closing)
                or (is_close and closing)
            )


            records.append({
                "frame": int(frame_number),
                "track_id_a": track_a,
                "class_a": first["class_name"],
                "track_id_b": track_b,
                "class_b": second["class_name"],
                "center_distance_px": center_distance,
                "normalized_center_distance": normalized_distance,
                "bbox_iou": iou,
                "closing_speed_px_per_sec": closing_speed,
                "is_close": bool(is_close),
                "boxes_overlap": bool(boxes_overlap),
                "is_closing": bool(closing),
                "candidate_score": score,
                "potential_collision_candidate": bool(candidate),
            })

    return pd.DataFrame(records)
