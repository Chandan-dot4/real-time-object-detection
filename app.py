"""
Main Web Application & REST API Server.
Powered by Flask. Provides real-time multipart video streaming, AJAX controls,
telemetry polling, model tuning, video uploads, and results export.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file
from werkzeug.utils import secure_filename

import config
from detector import detector
from video_processor import processor
from youtube_handler import youtube_handler

logger = logging.getLogger("App")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(config.STATIC_DIR / "uploads")
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    """Renders the main student project dashboard."""
    return render_template(
        "index.html",
        title=config.APP_TITLE,
        subtitle=config.APP_SUBTITLE,
        version=config.APP_VERSION,
        preset_videos=config.PRESET_YOUTUBE_VIDEOS,
        coco_classes=config.COCO_CLASSES,
        target_classes=config.TARGET_TRAFFIC_CLASSES,
        class_colors=config.CLASS_COLORS,
        default_conf=config.DEFAULT_CONF_THRESHOLD
    )

@app.route("/video_feed")
def video_feed():
    """Route providing real-time MJPEG stream to the web browser."""
    return Response(
        processor.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/start", methods=["POST"])
def api_start():
    """Starts video processing for the requested input source."""
    data = request.get_json(silent=True) or {}
    source_type = data.get("source_type", "sample")
    source_input = data.get("source_input", "")

    # Optional model and confidence overrides
    conf = float(data.get("conf_threshold", config.DEFAULT_CONF_THRESHOLD))
    processor.conf_threshold = conf
    
    classes = data.get("selected_classes")
    if classes and isinstance(classes, list):
        processor.selected_classes = classes

    model_choice = data.get("model_name", "yolo11n.pt")
    if model_choice != detector.model_name:
        detector.load_model(model_choice)

    logger.info("Starting detection. Source: %s | Target: %s", source_type, source_input)
    result = processor.start_source(source_type, source_input)
    return jsonify(result)

@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stops the active video processing pipeline."""
    processor.stop()
    return jsonify({"success": True, "message": "Video detection stopped."})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Toggles pause/resume state."""
    is_paused = processor.pause_toggle()
    return jsonify({"success": True, "is_paused": is_paused})

@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Dynamically modifies detection parameters in real time."""
    data = request.get_json(silent=True) or {}
    
    if "conf_threshold" in data:
        processor.conf_threshold = float(data["conf_threshold"])
    
    if "selected_classes" in data and isinstance(data["selected_classes"], list):
        processor.selected_classes = data["selected_classes"]
        
    if "model_name" in data and data["model_name"] != detector.model_name:
        detector.load_model(data["model_name"])

    return jsonify({"success": True, "message": "Detection settings updated."})

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns telemetry data for real-time dashboard UI updates."""
    state = processor.get_dashboard_state()
    return jsonify(state)

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Handles local MP4 / AVI video uploads."""
    if 'video_file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files['video_file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    logger.info("Uploaded video saved to: %s", save_path)

    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": save_path
    })

@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    """Starts recording the annotated video output."""
    res = processor.start_recording()
    return jsonify(res)

@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    """Stops recording and finalizes MP4 file."""
    res = processor.stop_recording()
    return jsonify(res)

@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """Exports and downloads all detection logs as a CSV file."""
    csv_file_path = processor.export_csv_logs()
    if os.path.exists(csv_file_path):
        return send_file(csv_file_path, as_attachment=True, download_name=os.path.basename(csv_file_path))
    return jsonify({"error": "Failed to generate CSV export"}), 500

@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Returns curated preset YouTube traffic streams."""
    return jsonify(config.PRESET_YOUTUBE_VIDEOS)

if __name__ == "__main__":
    logger.info("Launching Object Detection & Recognition Dashboard on http://%s:%d",
                config.SERVER_HOST, config.SERVER_PORT)
    # Start the Flask web application
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False, threaded=True)

