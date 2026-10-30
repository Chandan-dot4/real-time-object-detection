"""
Video Processing & Annotation Pipeline.
Manages video inputs (YouTube, local file, webcam, sample), frame extraction,
YOLO detection, object tracking, visual annotations (bounding boxes, IDs, trails, HUD),
real-time statistics, CSV logging, and video recording.
"""

import os
import cv2
import time
import math
import csv
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Generator
import numpy as np

import config
from detector import detector
from tracker import MultiObjectTracker
from youtube_handler import youtube_handler

logger = logging.getLogger("VideoProcessor")
logging.basicConfig(level=logging.INFO)

class VideoProcessor:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.tracker = MultiObjectTracker()
        
        # State Flags
        self.is_running = False
        self.is_paused = False
        self.lock = threading.Lock()
        
        # Source Information
        self.source_type = "none"   # 'youtube', 'local', 'webcam', 'sample'
        self.source_path = ""
        self.source_title = "No Video Loaded"
        self.total_frames = 0
        self.frame_index = 0
        self.frame_width = 1280
        self.frame_height = 720
        
        # Hyperparameters (Dynamic via UI controls)
        self.conf_threshold = config.DEFAULT_CONF_THRESHOLD
        self.iou_threshold = config.DEFAULT_IOU_THRESHOLD
        self.selected_classes = list(config.TARGET_TRAFFIC_CLASSES.keys())
        self.selected_model = "yolo11n.pt"
        
        # Metrics & Performance
        self.fps = 0.0
        self.inference_ms = 0.0
        self.last_frame_time = time.time()
        self.current_annotated_frame: Optional[np.ndarray] = None
        
        # Real-time Detection Event Logs
        self.detection_logs: List[Dict[str, Any]] = []
        self.max_logs = 150
        
        # Video Recording
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_filename = ""
        
        # Threading Worker
        self.processing_thread: Optional[threading.Thread] = None

    def start_source(self, source_type: str, source_input: str) -> Dict[str, Any]:
        """
        Initializes and launches video capture from YouTube, local file, webcam, or sample.
        """
        self.stop()

        with self.lock:
            self.tracker.reset()
            self.detection_logs.clear()
            self.source_type = source_type
            actual_stream_path = source_input
            title = "Video Stream"

            if source_type == "youtube":
                yt_res = youtube_handler.extract_direct_stream_url(source_input)
                if not yt_res["success"] or not yt_res["stream_url"]:
                    logger.warning("YouTube stream resolution failed: %s", yt_res.get("error"))
                    return {
                        "success": False,
                        "error": yt_res.get("error") or "Failed to resolve YouTube stream."
                    }
                actual_stream_path = yt_res["stream_url"]
                title = yt_res.get("title", "YouTube Traffic Video")

            elif source_type == "webcam":
                try:
                    cam_idx = int(source_input) if source_input.isdigit() else 0
                    actual_stream_path = cam_idx
                    title = f"Live Webcam (Device {cam_idx})"
                except Exception:
                    actual_stream_path = 0
                    title = "Live Webcam"

            elif source_type == "local":
                title = os.path.basename(source_input)
                if not os.path.exists(source_input):
                    return {"success": False, "error": f"Local file not found: {source_input}"}

            elif source_type == "sample":
                sample_path = config.SAMPLES_DIR / "traffic_sample.mp4"
                if not sample_path.exists():
                    return {"success": False, "error": "Sample traffic video not found."}
                actual_stream_path = str(sample_path)
                title = "Standard Traffic Benchmark (Intel IoT)"

            # Open Video Capture
            logger.info("Opening VideoCapture for: %s", title)
            cap = cv2.VideoCapture(actual_stream_path)

            # Apply buffer and timeout optimizations
            if source_type in ["youtube", "webcam"]:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                return {"success": False, "error": f"Failed to open video source: {title}"}

            self.cap = cap
            self.source_path = str(source_input)
            self.source_title = title
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.frame_index = 0
            self.is_running = True
            self.is_paused = False

            # Spawn dedicated video worker thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            logger.info("Video pipeline running successfully: %s (%dx%d)", title, self.frame_width, self.frame_height)
            return {
                "success": True,
                "title": title,
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "total_frames": self.total_frames
            }

    def _processing_loop(self):
        """Continuous background thread processing frames, running YOLO and tracking."""
        prev_time = time.time()
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.08)
                continue

            if not self.cap or not self.cap.isOpened():
                break

            ret, raw_frame = self.cap.read()

            if not ret or raw_frame is None or raw_frame.size == 0:
                # Loop local/sample videos for continuous presentation demo
                if self.source_type in ["local", "sample"]:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_index = 0
                    continue
                else:
                    logger.info("End of stream reached.")
                    break

            self.frame_index += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                # Exponential moving average for smooth FPS readout
                instant_fps = 1.0 / dt
                self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps

            # Normalize frame resolution to avoid CPU bottleneck on laptops (max 960 width)
            h, w = raw_frame.shape[:2]
            target_w = 960
            if w > target_w:
                scale = target_w / float(w)
                display_frame = cv2.resize(raw_frame, (target_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = raw_frame.copy()

            # Measure deep inference time
            t_start = time.time()
            detections = detector.detect(
                display_frame,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                allowed_classes=self.selected_classes
            )
            self.inference_ms = (time.time() - t_start) * 1000.0

            # Update Multi-Object Tracker
            active_tracks = self.tracker.update(detections)

            # Record detection log events
            timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
            for track in active_tracks:
                if track["confidence"] >= self.conf_threshold:
                    self._add_log_entry({
                        "time": timestamp_str,
                        "frame": self.frame_index,
                        "id": track["id"],
                        "class": track["class_name"].capitalize(),
                        "confidence": f"{track['confidence']*100:.1f}%",
                        "centroid": f"({track['centroid'][0]}, {track['centroid'][1]})"
                    })

            # Draw Annotations, Tracking IDs, Motion Trails, and Heads-Up Display
            annotated_frame = self._render_annotations(display_frame, active_tracks)

            # Write to video recorder if active
            if self.is_recording and self.video_writer:
                try:
                    self.video_writer.write(annotated_frame)
                except Exception as e:
                    logger.error("Error writing frame to recorder: %s", e)

            with self.lock:
                self.current_annotated_frame = annotated_frame

            # Throttle slightly to achieve natural playback speed (~25-30 FPS)
            target_delay = 0.033 - (time.time() - curr_time)
            if target_delay > 0.005:
                time.sleep(target_delay)

        logger.info("Video processing loop terminated.")
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _render_annotations(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws high-aesthetic, professional computer vision annotations:
        - Bounding boxes with corner accents
        - Label badge with Class, Confidence, and Unique Track ID
        - Object movement trails (centroid history)
        - Modern HUD status bar overlay
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw Object Trails and Bounding Boxes
        for trk in tracks:
            bbox = trk["bbox"]
            cls_name = trk["class_name"].lower()
            obj_id = trk["id"]
            conf = trk["confidence"]
            color = config.get_class_color(cls_name)

            x1, y1, x2, y2 = bbox
            # Clamp to frame boundary
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # 1. Draw motion path (trajectory history line)
            history = trk.get("history", [])
            if len(history) > 1:
                pts = np.array(history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                # Centroid dot
                cv2.circle(canvas, trk["centroid"], 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # 2. Main Bounding Box (Clean sleek outline)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 3. Corner Accents (Targeting reticle aesthetic)
            corner_len = min(16, int((x2 - x1) / 4), int((y2 - y1) / 4))
            thick = 3
            if corner_len > 4:
                # Top-Left
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thick)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thick)
                # Top-Right
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), thick)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), thick)
                # Bottom-Left
                cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), thick)
                cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), thick)
                # Bottom-Right
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), thick)
                cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), thick)

            # 4. Label Badge above Bounding Box: [ID: #X] ClassName Conf%
            label = f"ID #{obj_id} | {cls_name.upper()} {conf*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - label_h - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w - 1, x1 + label_w + 10)

            # Solid color badge with dark background
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 24, 30), -1)
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.putText(canvas, label, (badge_x1 + 5, badge_y2 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 5. Heads-Up Display (HUD) Status Bar (Top bar)
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 36), (15, 20, 26), -1)
        # Apply 70% opacity blending
        cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0, canvas)

        stats = self.tracker.get_statistics()
        active_cnt = stats["active_objects"]
        total_veh = stats["cumulative_breakdown"]["total_vehicles"]
        peds = stats["cumulative_breakdown"]["person"]

        # Left HUD: Mode and Source
        hud_text_left = f"SRC: {self.source_type.upper()} | {self.source_title[:28]}"
        cv2.putText(canvas, hud_text_left, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 220, 240), 1, cv2.LINE_AA)

        # Right HUD: Live FPS, Latency, Active & Cumulative Counts
        hud_text_right = f"FPS: {self.fps:.1f} | LATENCY: {self.inference_ms:.1f}ms | ACTIVE: {active_cnt} | VEHICLES: {total_veh} | PEDESTRIANS: {peds}"
        (r_w, _), _ = cv2.getTextSize(hud_text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(canvas, hud_text_right, (max(10, w - r_w - 12), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 160), 1, cv2.LINE_AA)

        return canvas

    def _add_log_entry(self, entry: Dict[str, Any]):
        """Adds a detection event log entry in memory."""
        self.detection_logs.append(entry)
        if len(self.detection_logs) > self.max_logs:
            self.detection_logs.pop(0)

    def pause_toggle(self) -> bool:
        """Toggles between paused and running states."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def stop(self):
        """Stops the current video stream and cleans up resources."""
        self.is_running = False
        self.is_paused = False
        if self.is_recording:
            self.stop_recording()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def start_recording(self) -> Dict[str, Any]:
        """Initializes MP4 video recording for the annotated output."""
        if self.is_recording:
            return {"success": False, "error": "Recording is already active."}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"detection_record_{timestamp}.mp4"
        out_path = config.RECORDINGS_DIR / self.recording_filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = max(15.0, min(30.0, self.fps if self.fps > 0 else 24.0))
        width = 960 if self.frame_width > 960 else self.frame_width
        height = int(self.frame_height * (width / float(self.frame_width)))

        self.video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not self.video_writer.isOpened():
            return {"success": False, "error": "Could not open video writer codec."}

        self.is_recording = True
        logger.info("Started recording annotated stream to: %s", out_path)
        return {"success": True, "filename": self.recording_filename}

    def stop_recording(self) -> Dict[str, Any]:
        """Finalizes and flushes the recorded video file."""
        if not self.is_recording:
            return {"success": False, "error": "No recording in progress."}

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        logger.info("Saved recording: %s", self.recording_filename)
        return {"success": True, "filename": self.recording_filename}

    def export_csv_logs(self) -> str:
        """Exports all detection and tracking events to a CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"object_detections_{timestamp}.csv"
        csv_path = config.EXPORTS_DIR / csv_filename

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Frame Number", "Track ID", "Class Name", "Confidence", "Centroid (X,Y)"])
            for log in self.detection_logs:
                writer.writerow([
                    log.get("time", ""),
                    log.get("frame", ""),
                    log.get("id", ""),
                    log.get("class", ""),
                    log.get("confidence", ""),
                    log.get("centroid", "")
                ])

        return str(csv_path)

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Provides full real-time telemetry and statistics for the dashboard UI."""
        stats = self.tracker.get_statistics()
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_recording": self.is_recording,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "fps": round(self.fps, 1),
            "inference_ms": round(self.inference_ms, 1),
            "frame_index": self.frame_index,
            "total_frames": self.total_frames,
            "active_objects": stats["active_objects"],
            "total_cumulative": stats["total_cumulative"],
            "active_counts": stats["active_breakdown"],
            "cumulative_counts": stats["cumulative_breakdown"],
            "logs": list(reversed(self.detection_logs[-25:]))  # Most recent 25 events
        }

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Generator producing multipart/x-mixed-replace JPEG frames for the web browser.
        """
        while True:
            frame_to_encode = None
            with self.lock:
                if self.current_annotated_frame is not None:
                    frame_to_encode = self.current_annotated_frame.copy()

            if frame_to_encode is None:
                # Standby / Idle screen
                idle_canvas = np.zeros((480, 854, 3), dtype=np.uint8)
                idle_canvas[:] = (20, 24, 32)
                # Standby graphics
                cv2.putText(idle_canvas, "OBJECT DETECTION SYSTEM READY", (180, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(idle_canvas, "Select a Video Source or click 'Start Detection' to begin", (150, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 180, 200), 1, cv2.LINE_AA)
                cv2.rectangle(idle_canvas, (120, 150), (734, 320), (50, 65, 85), 2)
                frame_to_encode = idle_canvas
            elif self.is_paused:
                # Add PAUSED badge
                cv2.putText(frame_to_encode, "[PAUSED]", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # Compress as JPEG for browser streaming
            ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS streaming yield

# Global Video Processor Instance
processor = VideoProcessor()

