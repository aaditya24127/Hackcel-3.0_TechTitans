"""
Flask Application Entry Point
Exposes web routes, MJPEG video streaming, and REST APIs for camera control,
video uploads, live detection status, and database history.
"""

import os
import sys
import logging
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from database import test_connection, get_incoming_vehicles
from detection import DetectionPipeline, VideoProcessingWorker

load_dotenv()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("parking_app")

# Initialize Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "parking-component-secret-key")

# Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

# Initialize Computer Vision Pipeline & Worker
pipeline = DetectionPipeline()
worker = VideoProcessingWorker(pipeline)

def allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has a supported video or image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename: str) -> bool:
    """Checks if the filename has an image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in IMAGE_EXTENSIONS

@app.route("/")
def index():
    """Renders the main dashboard."""
    return render_template("dashboard.html")

def generate_frames():
    """Generator function yielding MJPEG video stream frames."""
    import time
    while True:
        frame_bytes = worker.get_jpeg_frame()
        if frame_bytes:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)  # ~30 FPS streaming rate

@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint for the dashboard video display."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    """Starts video acquisition and detection from a local camera source."""
    data = request.get_json() or {}
    camera_index = int(data.get("camera_index", 0))
    
    try:
        worker.start_camera(camera_index=camera_index)
        return jsonify({
            "status": "success",
            "message": f"Camera {camera_index} started successfully."
        })
    except Exception as e:
        logger.error(f"Failed to start camera {camera_index}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/camera/stop", methods=["POST"])
def stop_processing():
    """Stops the active camera, video, or image processing cleanly."""
    try:
        worker.stop()
        return jsonify({
            "status": "success",
            "message": "Processing stopped successfully."
        })
    except Exception as e:
        logger.error(f"Failed to stop processing: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/video/upload", methods=["POST"])
@app.route("/api/file/upload", methods=["POST"])
def upload_file():
    """Handles video or image file upload to the uploads/ directory."""
    file = None
    if "video" in request.files:
        file = request.files["video"]
    elif "file" in request.files:
        file = request.files["file"]
    elif "image" in request.files:
        file = request.files["image"]

    if file is None or file.filename == "":
        return jsonify({"status": "error", "message": "No media file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        media_type = "image" if is_image_file(filename) else "video"
        logger.info(f"{media_type.capitalize()} uploaded successfully: {save_path}")
        return jsonify({
            "status": "success",
            "filename": filename,
            "media_type": media_type,
            "file_path": save_path,
            "message": f"{media_type.capitalize()} uploaded successfully."
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"Unsupported format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

@app.route("/api/video/process", methods=["POST"])
@app.route("/api/file/process", methods=["POST"])
def process_file():
    """Starts processing an uploaded video or image file."""
    data = request.get_json() or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"status": "error", "message": "Filename not provided."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found on server."}), 404

    try:
        if is_image_file(filename):
            worker.start_image(file_path)
            return jsonify({
                "status": "success",
                "media_type": "image",
                "message": f"Processed image '{filename}'."
            })
        else:
            worker.start_video(file_path)
            return jsonify({
                "status": "success",
                "media_type": "video",
                "message": f"Started processing video '{filename}'."
            })
    except Exception as e:
        logger.error(f"Failed to process file '{filename}': {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns the current worker and detection telemetry."""
    summary = worker.get_status_summary()
    return jsonify(summary)

@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns recent incoming vehicle detection history from MySQL."""
    limit = request.args.get("limit", 50, type=int)
    history = get_incoming_vehicles(limit=limit)
    return jsonify({
        "status": "success",
        "count": len(history),
        "history": history
    })

@app.route("/api/db_status", methods=["GET"])
def get_db_status():
    """Checks MySQL connectivity."""
    connected, msg = test_connection()
    return jsonify({
        "connected": connected,
        "message": msg
    })

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    
    logger.info(f"Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
