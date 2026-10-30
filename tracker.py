"""
Real-Time Multi-Object Tracker (Centroid & IoU Association).
Assigns persistent unique IDs across video frames, tracks object motion paths
(movement trails), and provides cumulative and active vehicle/pedestrian counting.
"""

import math
from collections import OrderedDict
from typing import List, Dict, Any, Tuple
import numpy as np
import config

def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes: [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class TrackedObject:
    """Represents a single detected entity with a persistent trajectory."""
    def __init__(self, object_id: int, class_name: str, bbox: List[int], confidence: float):
        self.object_id = object_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.centroid = self._compute_centroid(bbox)
        self.history = [self.centroid]
        self.disappeared_count = 0
        self.counted = False

    @staticmethod
    def _compute_centroid(bbox: List[int]) -> Tuple[int, int]:
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def update(self, bbox: List[int], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.centroid = self._compute_centroid(bbox)
        self.history.append(self.centroid)
        if len(self.history) > 30:  # Retain trailing motion path up to 30 frames
            self.history.pop(0)
        self.disappeared_count = 0

class MultiObjectTracker:
    def __init__(self, max_disappeared: int = config.TRACKER_MAX_DISAPPEARED,
                 max_distance: float = config.TRACKER_MAX_DISTANCE,
                 min_iou: float = config.TRACKER_IOU_MIN):
        self.next_object_id = 1
        self.objects: Dict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou

        # Cumulative counters across video runtime
        self.total_unique_objects = 0
        self.cumulative_counts = {
            "car": 0,
            "bus": 0,
            "truck": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "person": 0,
            "traffic light": 0,
            "total_vehicles": 0,
            "other": 0
        }

    def reset(self):
        """Resets the tracker for a new video feed."""
        self.next_object_id = 1
        self.objects.clear()
        self.total_unique_objects = 0
        for k in self.cumulative_counts:
            self.cumulative_counts[k] = 0

    def register(self, class_name: str, bbox: List[int], confidence: float) -> int:
        """Registers a new object and increments class-specific counters."""
        new_id = self.next_object_id
        self.objects[new_id] = TrackedObject(new_id, class_name, bbox, confidence)
        self.next_object_id += 1
        self.total_unique_objects += 1

        # Increment cumulative stats
        cls_lower = class_name.lower()
        if cls_lower in self.cumulative_counts:
            self.cumulative_counts[cls_lower] += 1
        else:
            self.cumulative_counts["other"] += 1

        if cls_lower in ["car", "bus", "truck", "motorcycle", "bicycle"]:
            self.cumulative_counts["total_vehicles"] += 1

        return new_id

    def deregister(self, object_id: int):
        """Removes an object that has left the camera frame."""
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates object trajectories using a hybrid Euclidean + IoU Hungarian matching.
        
        Args:
            detections: list of dicts with keys: 'bbox' [x1, y1, x2, y2], 'class_name', 'confidence'
            
        Returns:
            list of active tracked objects with IDs and trajectories
        """
        # Case 1: No detections in current frame
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared_count += 1
                if self.objects[object_id].disappeared_count > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_active_tracks()

        # Case 2: No currently tracked objects -> register all incoming
        if len(self.objects) == 0:
            for det in detections:
                self.register(det["class_name"], det["bbox"], det["confidence"])
            return self.get_active_tracks()

        # Case 3: Match existing objects with new detections
        object_ids = list(self.objects.keys())
        existing_tracks = [self.objects[oid] for oid in object_ids]

        # Calculate cost matrices (distance and IoU)
        D = np.zeros((len(existing_tracks), len(detections)), dtype="float")
        IOU = np.zeros((len(existing_tracks), len(detections)), dtype="float")

        for i, track in enumerate(existing_tracks):
            for j, det in enumerate(detections):
                det_center = ((det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2)
                D[i, j] = math.hypot(track.centroid[0] - det_center[0], track.centroid[1] - det_center[1])
                IOU[i, j] = calculate_iou(track.bbox, det["bbox"])

        # Prioritize matching by smallest Euclidean distance and high IoU
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue

            # Check distance threshold or IoU threshold
            distance = D[row, col]
            iou_val = IOU[row, col]
            same_class = (existing_tracks[row].class_name.lower() == detections[col]["class_name"].lower())

            # Validate association
            if (distance <= self.max_distance or iou_val >= self.min_iou) and (same_class or distance <= 45.0):
                oid = object_ids[row]
                self.objects[oid].update(detections[col]["bbox"], detections[col]["confidence"])
                assigned_rows.add(row)
                assigned_cols.add(col)

        # Handle unassigned existing tracks (disappeared)
        unassigned_rows = set(range(len(existing_tracks))) - assigned_rows
        for row in unassigned_rows:
            oid = object_ids[row]
            self.objects[oid].disappeared_count += 1
            if self.objects[oid].disappeared_count > self.max_disappeared:
                self.deregister(oid)

        # Handle unassigned new detections (new objects entered)
        unassigned_cols = set(range(len(detections))) - assigned_cols
        for col in unassigned_cols:
            self.register(detections[col]["class_name"], detections[col]["bbox"], detections[col]["confidence"])

        return self.get_active_tracks()

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Returns the current snapshot of active tracked objects for rendering."""
        active = []
        for oid, track in self.objects.items():
            active.append({
                "id": oid,
                "class_name": track.class_name,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "centroid": track.centroid,
                "history": list(track.history)
            })
        return active

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates real-time active and cumulative statistics for the dashboard."""
        active_counts = {
            "car": 0,
            "bus": 0,
            "truck": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "person": 0,
            "traffic light": 0,
            "total_vehicles": 0,
            "other": 0
        }

        for track in self.objects.values():
            c = track.class_name.lower()
            if c in active_counts:
                active_counts[c] += 1
            else:
                active_counts["other"] += 1

            if c in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                active_counts["total_vehicles"] += 1

        return {
            "active_objects": len(self.objects),
            "total_cumulative": self.total_unique_objects,
            "active_breakdown": active_counts,
            "cumulative_breakdown": dict(self.cumulative_counts)
        }

