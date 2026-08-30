import time
import threading
import cv2
from modules.M1_detection.peoplenet_infer import PeopleNetDetector
from modules.M1_detection.draw_utils import Visualizer


class StreamManager:
    def __init__(self, source, model_path: str):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.detector = PeopleNetDetector(model_path=model_path)
        
        self.latest_frame = None
        self.latest_detections = []
        self.current_fps = 0.0
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Starts worker thread for continuous frame capture and inference."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        prev_time = time.time()

        while self.is_running:
            success, frame = self.cap.read()
            if not success:
                # Loop video source if playing static test video files
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            # Run detection inference
            detections = self.detector.detect(frame)

            # Compute real-time FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Render overlay
            annotated = Visualizer.draw_detections(frame, detections, fps=fps)

            # Thread-safe write
            with self.lock:
                self.latest_frame = annotated
                self.latest_detections = detections
                self.current_fps = fps

    def get_jpeg_frame(self):
        """Encodes annotated frame as JPEG for Flask MJPEG route streaming."""
        with self.lock:
            if self.latest_frame is None:
                return None
            success, buffer = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes() if success else None

    def get_current_stats(self):
        """Returns JSON-serializable stats for frontend metrics."""
        with self.lock:
            person_count = sum(1 for d in self.latest_detections if d["label"] == "person")
            return {
                "fps": round(self.current_fps, 1),
                "person_count": person_count,
                "total_detections": len(self.latest_detections),
                "detections": self.latest_detections
            }

    def stop(self):
        """Stops the thread and releases the video stream."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()