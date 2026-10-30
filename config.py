"""
Configuration Module for Real-Time Object Detection and Recognition System.
Defines supported classes, color palettes, model parameters, video defaults,
and application settings optimized for student presentations and laptop hardware.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "sample_videos"
RECORDINGS_DIR = BASE_DIR / "recordings"
EXPORTS_DIR = BASE_DIR / "exports"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
for path in [STATIC_DIR, SAMPLES_DIR, RECORDINGS_DIR, EXPORTS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_TITLE = "Real-Time Object Detection and Recognition System"
APP_SUBTITLE = "Computer Vision & Deep Learning Traffic Surveillance"
APP_VERSION = "2.4.0 (College Group Project Edition)"

# Supported COCO Classes (80 Classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Primary Target Categories for Traffic and Pedestrian Monitoring
TARGET_TRAFFIC_CLASSES = {
    "car": "Vehicle",
    "bus": "Vehicle",
    "truck": "Vehicle",
    "motorcycle": "Two-Wheeler",
    "bicycle": "Two-Wheeler",
    "person": "Pedestrian",
    "traffic light": "Infrastructure"
}

# Color Mapping for Aesthetic Bounding Boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "car": (255, 128, 0),        # Vibrant Blue/Orange
    "bus": (0, 200, 255),        # Amber / Gold
    "truck": (50, 160, 255),      # Light Coral
    "motorcycle": (255, 50, 150), # Purple / Magenta
    "bicycle": (0, 220, 120),    # Emerald Green
    "person": (30, 255, 100),    # Mint Green
    "traffic light": (0, 240, 255),# Bright Yellow
    "default": (200, 200, 200)   # Clean Slate Grey
}

def get_class_color(class_name: str) -> tuple:
    """Returns a designated BGR color tuple for a class name."""
    return CLASS_COLORS.get(class_name.lower(), CLASS_COLORS["default"])

# Object Detection Hyperparameters
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MODEL_SIZE = 640  # 640x640 standard YOLO input size

# Object Tracking Parameters (Centroid + IoU Kalman-like tracker)
TRACKER_MAX_DISAPPEARED = 25  # Frames to retain identity when temporarily occluded
TRACKER_MAX_DISTANCE = 85.0   # Maximum pixel distance to associate object across frames
TRACKER_IOU_MIN = 0.25        # Minimum IoU for bounding box association

# Hardware Optimization
INFERENCE_DEVICE = "cpu"      # 'cpu' or 'cuda' (auto-selected in detector)
ENABLE_FRAME_SKIPPING = True  # Run heavy deep inference every N frames if CPU load is high
DEFAULT_SKIP_FRAMES = 1       # 1 = process every frame; 2 = process alternate frames

# Curated YouTube Traffic Demonstration Videos (Valid & active video URLs)
PRESET_YOUTUBE_VIDEOS = [
    {
        "name": "Busy Urban Traffic & Crosswalk (Tokyo/Seoul 4K)",
        "url": "https://www.youtube.com/watch?v=1EiC9bvVGnk",
        "description": "High density multi-lane traffic with cars, buses, bikes, and crossing pedestrians."
    },
    {
        "name": "Times Square Pedestrian & Vehicle Flow (NYC Live Cam)",
        "url": "https://www.youtube.com/watch?v=1-iS7LArMPA",
        "description": "Dense pedestrian stream, yellow cabs, buses, and delivery vans."
    },
    {
        "name": "Highway Traffic Surveillance (Full HD)",
        "url": "https://www.youtube.com/watch?v=KBsqQez-O4w",
        "description": "Fast moving highway vehicles, trucks, lane changes, and vehicle speed tracking."
    }
]

# Web Dashboard Server Settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000

