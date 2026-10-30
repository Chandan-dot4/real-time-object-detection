"""
Object Detection Engine using Ultralytics YOLO (YOLOv8 / YOLO11).
Handles deep learning inference, non-maximum suppression (NMS), confidence filtering,
class-specific filtering, and hardware acceleration (CPU / GPU).
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import config

logger = logging.getLogger("YOLODetector")
logging.basicConfig(level=logging.INFO)

# Suppress telemetry and offline notices for clean console output
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "false"

class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt", device: Optional[str] = None):
        """
        Initializes the YOLO detection engine.
        
        Args:
            model_name: 'yolo11n.pt' or 'yolov8n.pt'
            device: 'cpu', 'cuda', or None for auto-selection
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = config.COCO_CLASSES
        self.is_loaded = False
        
        self.load_model(model_name)

    def load_model(self, model_name: str) -> bool:
        """Loads or reloads the chosen YOLO model checkpoint."""
        try:
            from ultralytics import YOLO

            # Resolve model weights path
            model_path = config.MODELS_DIR / model_name
            if not model_path.exists():
                logger.warning("Local weights not found at %s. Attempting auto-fetch...", model_path)
                model_ref = model_name
            else:
                model_ref = str(model_path)

            logger.info("Loading YOLO model: %s on device: %s", model_ref, self.device)
            self.model = YOLO(model_ref)
            
            # Extract class names mapping from model if available
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]

            self.model_name = model_name
            self.is_loaded = True
            logger.info("YOLO model initialized successfully. Total classes: %d", len(self.class_names))
            return True

        except Exception as e:
            logger.error("Failed to load YOLO model %s: %s", model_name, e)
            self.is_loaded = False
            return False

    def detect(self, frame: np.ndarray, conf_threshold: float = config.DEFAULT_CONF_THRESHOLD,
               iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
               allowed_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Executes real-time object detection on an input BGR frame.
        
        Args:
            frame: OpenCV BGR image (numpy ndarray)
            conf_threshold: minimum detection confidence (0.0 to 1.0)
            iou_threshold: NMS IoU threshold
            allowed_classes: list of class names to filter, or None for all
            
        Returns:
            List of detected object dictionaries with 'bbox', 'class_name', 'confidence', 'class_id'
        """
        if not self.is_loaded or self.model is None or frame is None or frame.size == 0:
            return []

        clean_allowed = [c.lower() for c in allowed_classes] if allowed_classes else None
        detections = []

        try:
            # Perform inference with gradients disabled for maximum FPS
            with torch.no_grad():
                results = self.model(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False,
                    imgsz=config.DEFAULT_MODEL_SIZE
                )

            if not results or len(results) == 0:
                return []

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return []

            # Extract coordinates, confidences, and class IDs
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cid = clss[i]
                cname = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                cname_lower = cname.lower()

                # Filter by user-selected classes
                if clean_allowed and cname_lower not in clean_allowed:
                    continue

                conf = float(confs[i])
                box = [int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])]

                detections.append({
                    "bbox": box,
                    "class_name": cname,
                    "confidence": conf,
                    "class_id": cid
                })

        except Exception as e:
            logger.error("Error during YOLO detection inference: %s", e)

        return detections

# Global Detector Instance (YOLO11n default, fallback to YOLOv8n)
default_model = "yolo11n.pt" if (config.MODELS_DIR / "yolo11n.pt").exists() else "yolov8n.pt"
detector = ObjectDetector(model_name=default_model)

