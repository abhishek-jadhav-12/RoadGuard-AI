
from pathlib import Path
from ultralytics import YOLO


class RoadObjectTracker:
    """Track road objects across video frames using ByteTrack."""

    def __init__(self, model_path="yolo11n.pt", tracker="bytetrack.yaml"):
        self.model = YOLO(str(Path(model_path)))
        self.tracker = tracker

    def track_video(
        self,
        video_path,
        confidence=0.40,
        image_size=640,
        save=True,
        output_dir="outputs/detections",
    ):
        """Run persistent multi-object tracking on a video."""
        return self.model.track(
            source=str(video_path),
            conf=confidence,
            imgsz=image_size,
            tracker=self.tracker,
            persist=True,
            stream=True,
            save=save,
            project=str(output_dir),
            name="tracked_video",
            exist_ok=True,
            verbose=False,
        )