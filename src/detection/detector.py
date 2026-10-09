
from pathlib import Path

from ultralytics import YOLO


class RoadObjectDetector:
    """YOLO-based road object detector."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        source,
        confidence: float = 0.40,
        image_size: int = 640,
        **kwargs,
    ):
        """Run YOLO inference on an image, video, or supported source."""
        return self.model.predict(
            source=source,
            conf=confidence,
            imgsz=image_size,
            verbose=False,
            **kwargs,
        )