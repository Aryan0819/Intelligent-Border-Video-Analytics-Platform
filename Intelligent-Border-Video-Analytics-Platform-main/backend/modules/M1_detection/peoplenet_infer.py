import cv2
import numpy as np
import onnxruntime as ort
from .preprocessor import ImagePreprocessor


class PeopleNetDetector:
    # Standard classes supported by NVIDIA PeopleNet
    CLASSES = ["person", "bag", "face"]

    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.preprocessor = ImagePreprocessor(target_shape=(544, 960))

        # Initialize ONNX Runtime Session (auto-selects CUDA if available, else CPU)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _postprocess(self, outputs, meta):
        """
        Parses raw model output tensors into absolute unpadded bounding boxes.
        """
        boxes, scores, class_ids = [], [], []
        scale, pad_left, pad_top = meta["scale"], meta["pad_left"], meta["pad_top"]
        orig_h, orig_w = meta["orig_shape"]

        # Parse DetectNet_v2 GridBox tensors: outputs -> [coverage, bboxes]
        if len(outputs) == 2:
            cov, bbox = outputs[0], outputs[1]
            # Handle shape variants (cov: [1, num_classes, H_grid, W_grid])
            if cov.ndim == 4:
                _, num_classes, gh, gw = cov.shape
                stride_x = 960 / gw
                stride_y = 544 / gh

                for c in range(num_classes):
                    for y in range(gh):
                        for x in range(gw):
                            confidence = float(cov[0, c, y, x])
                            if confidence > self.conf_threshold:
                                # Grid cell offsets
                                mx = x * stride_x
                                my = y * stride_y
                                
                                dx1, dy1, dx2, dy2 = bbox[0, c*4:(c+1)*4, y, x]
                                bx1 = mx - dx1
                                by1 = my - dy1
                                bx2 = mx + dx2
                                by2 = my + dy2

                                # Remap from padded coordinate space to original frame dimensions
                                x1 = max(0, int((bx1 - pad_left) / scale))
                                y1 = max(0, int((by1 - pad_top) / scale))
                                x2 = min(orig_w, int((bx2 - pad_left) / scale))
                                y2 = min(orig_h, int((by2 - pad_top) / scale))

                                w = max(0, x2 - x1)
                                h = max(0, y2 - y1)

                                boxes.append([x1, y1, w, h])
                                scores.append(confidence)
                                class_ids.append(c)

        # Apply OpenCV NMS
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
            for idx in indices:
                i = idx if isinstance(idx, (int, np.integer)) else idx[0]
                bx, by, bw, bh = boxes[i]
                detections.append({
                    "box": [bx, by, bx + bw, by + bh], # [x1, y1, x2, y2]
                    "confidence": float(scores[i]),
                    "class_id": class_ids[i],
                    "label": self.CLASSES[class_ids[i]] if class_ids[i] < len(self.CLASSES) else "unknown"
                })

        return detections

    def detect(self, frame: np.ndarray):
        """
        Main interface method: Input raw frame -> Return detected objects.
        """
        tensor, meta = self.preprocessor.preprocess(frame)
        raw_outputs = self.session.run(None, {self.input_name: tensor})
        return self._postprocess(raw_outputs, meta)