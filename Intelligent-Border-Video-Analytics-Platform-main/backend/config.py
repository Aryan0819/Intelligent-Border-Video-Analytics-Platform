import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Model Configuration
MODEL_PATH = os.path.join(BASE_DIR, "models", "peoplenet", "resnet34_peoplenet.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "models", "peoplenet", "labels.txt")

# Inference Parameters
INPUT_WIDTH = 960
INPUT_HEIGHT = 544
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Target Video Source (Use 0 for webcam, or path to MP4 file)
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "data/input_videos/sample_cctv.mp4")

# Server Configuration
HOST = "0.0.0.0"
PORT = 5000
