

import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, target_shape=(544, 960)):
        """
        :param target_shape: Target input dimensions (height, width).
                             Default 544x960 is the native resolution for NVIDIA PeopleNet.
        """
        self.target_height, self.target_width = target_shape

    def preprocess(self, frame: np.ndarray):
        """
        Transforms raw frame to normalized CHW tensor and returns scaling metadata.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Compute scale and padding (letterboxing)
        scale = min(self.target_width / orig_w, self.target_height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.target_width - new_w
        pad_h = self.target_height - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        # 2. Add border padding (constant zero padding)
        padded_frame = cv2.copyMakeBorder(
            resized_frame,
            pad_top, pad_h - pad_top,
            pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # 3. BGR to RGB -> Normalization [0, 1] -> HWC to CHW -> Add Batch Dimension
        rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)

        meta = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "orig_shape": (orig_h, orig_w)
        }

        return tensor, meta