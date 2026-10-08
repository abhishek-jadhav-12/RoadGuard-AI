from pathlib import Path

from ultralytics import YOLO


class RoadObjectDetector:
    """YOLO-based road object detector."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def predict(self, source, confidence: float = 0.40):
        """Run object detection on an image/video source."""

        return self.model.predict(
            source=source,
            conf=confidence,
            verbose=False,
        )