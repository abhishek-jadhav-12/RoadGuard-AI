
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SeverityAssessment:
    level: str
    score: float
    confidence: str
    reasons: list[str]
    limitations: list[str]

    def to_dict(self):
        return asdict(self)


def estimate_severity(
    bbox_iou: float,
    closing_speed_px_s: Optional[float],
    sudden_motion_change: bool = False,
    involved_objects: int = 2,
    track_reliable: bool = False,
) -> SeverityAssessment:
    """
    Produce a preliminary visual-risk estimate.

    This is a heuristic prototype, not a validated injury or
    vehicle-damage severity model. Pixel motion is not physical speed.
    """
    score = 0.0
    reasons = []

    if bbox_iou >= 0.70:
        score += 0.35
        reasons.append("Very high bounding-box overlap")
    elif bbox_iou >= 0.20:
        score += 0.20
        reasons.append("Moderate bounding-box overlap")

    if closing_speed_px_s is not None:
        if closing_speed_px_s >= 50:
            score += 0.25
            reasons.append("High image-space closing speed")
        elif closing_speed_px_s >= 10:
            score += 0.15
            reasons.append("Moderate image-space closing speed")
    else:
        reasons.append("Closing-speed evidence unavailable")

    if sudden_motion_change:
        score += 0.20
        reasons.append("Sudden trajectory or motion change")

    if involved_objects >= 3:
        score += 0.10
        reasons.append("Three or more tracked objects involved")

    if track_reliable:
        score += 0.10
        reasons.append("Tracking reliability criterion satisfied")
    else:
        reasons.append("Tracking reliability not established")

    score = min(score, 1.0)

    # Conservative labels for visual review only.
    if score >= 0.65:
        level = "HIGH_VISUAL_RISK"
    elif score >= 0.35:
        level = "MODERATE_VISUAL_RISK"
    else:
        level = "LOW_VISUAL_RISK"

    confidence = (
        "limited"
        if closing_speed_px_s is None or not track_reliable
        else "provisional"
    )

    return SeverityAssessment(
        level=level,
        score=round(score, 3),
        confidence=confidence,
        reasons=reasons,
        limitations=[
            "Not validated against ground-truth severity labels",
            "Pixel measurements are not real-world speed or impact force",
            "Does not estimate injuries or vehicle damage",
            "Requires human review; not for emergency triage",
        ],
    )
