import cv2
import numpy as np


class Visualizer:
    # Color palette for classes (BGR)
    COLORS = {
        "person": (0, 255, 127),  # High-visibility Green
        "bag": (255, 165, 0),     # Cyan/Orange
        "face": (255, 215, 0)     # Light Gold
    }

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list, fps: float = 0.0) -> np.ndarray:
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]
            color = Visualizer.COLORS.get(label, (0, 255, 0))

            # 1. Draw corner/bounding rect
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # 2. Draw label banner
            caption = f"{label.upper()} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Keep label inside top bounds
            top_y = max(y1, text_h + 10)
            cv2.rectangle(
                annotated_frame, 
                (x1, top_y - text_h - 6), 
                (x1 + text_w + 6, top_y + baseline - 2), 
                color, 
                -1
            )
            cv2.putText(
                annotated_frame, 
                caption, 
                (x1 + 3, top_y - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1, 
                cv2.LINE_AA
            )

        # 3. Draw Telemetry Overlay (FPS & Total Humans Detected)
        human_count = sum(1 for d in detections if d["label"] == "person")
        status_text = f"FPS: {fps:.1f} | People: {human_count}"
        
        cv2.putText(
            annotated_frame, 
            status_text, 
            (15, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2, 
            cv2.LINE_AA
        )

        return annotated_frame