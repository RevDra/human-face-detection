"""
Flask Web Application for YOLOv12 Human Face Detection
Provides REST APIs and Web UI for image, video, and webcam detection
"""

import base64
import gc
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from werkzeug.utils import secure_filename

from face_detector import YOLOv12FaceDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
TEMPLATE_FOLDER = BASE_DIR / "web" / "templates"
STATIC_FOLDER = BASE_DIR / "web" / "static"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_FOLDER),
    static_folder=str(STATIC_FOLDER),
)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)

# Model registry
AVAILABLE_MODELS = {
    "yolov12n": "models/yolov12n-face.onnx",
    "yolov12s": "models/yolov12s-face.onnx",
    "yolov12m": "models/yolov12m-face.onnx",
    "yolov12l": "models/yolov12l-face.onnx",
}

MAX_LOADED_MODELS = int(os.environ.get("MAX_LOADED_MODELS", 2))
detectors: "OrderedDict[str, YOLOv12FaceDetector]" = OrderedDict()
detectors_lock = threading.Lock()

video_jobs = {}
video_jobs_lock = threading.Lock()


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def cleanup_old_files(max_age_seconds: int = 3600):
    """Remove temporary uploaded and output files older than max_age_seconds."""
    now = time.time()
    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        try:
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.name != ".gitkeep":
                    if now - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Cleanup warning in {folder}: {e}")

    with video_jobs_lock:
        expired_jobs = [
            jid
            for jid, job in video_jobs.items()
            if now - job.get("created_at", now) > max_age_seconds
        ]
        for jid in expired_jobs:
            video_jobs.pop(jid, None)


def get_detector(
    conf_threshold: float = 0.32, model_key: str = "yolov12n"
) -> YOLOv12FaceDetector:
    """
    Get or initialize a YOLOv12FaceDetector instance with LRU memory eviction.
    """
    if model_key not in AVAILABLE_MODELS:
        model_key = "yolov12n"

    with detectors_lock:
        if model_key in detectors:
            detectors.move_to_end(model_key)
            detector = detectors[model_key]
            detector.conf_threshold = conf_threshold
            return detector

        while len(detectors) >= MAX_LOADED_MODELS:
            evicted_key, _ = detectors.popitem(last=False)
            logger.info(f"Evicted model '{evicted_key}' from LRU cache to free RAM.")
            gc.collect()

        model_path = AVAILABLE_MODELS[model_key]
        logger.info(f"Loading model '{model_key}' from {model_path}...")
        detector = YOLOv12FaceDetector(
            model_path=model_path, conf_threshold=conf_threshold
        )
        detectors[model_key] = detector
        return detector


@app.route("/")
def index():
    """Render main web interface."""
    return render_template("index.html")

@app.route("/api/models", methods=["GET"])
def get_models():
    """Return available YOLOv12 face detection models for frontend selectors."""
    models_info = {
        "nano": {
            "name": "yolov12n",
            "label": "YOLOv12 Nano (Fastest - Realtime)",
            "description": "Optimized for speed, ideal for webcam."
        },
        "small": {
            "name": "yolov12s",
            "label": "YOLOv12 Small (Balanced)",
            "description": "Good balance between speed and accuracy."
        },
        "medium": {
            "name": "yolov12m",
            "label": "YOLOv12 Medium (High Accuracy)",
            "description": "Higher precision for video analysis."
        },
        "large": {
            "name": "yolov12l",
            "label": "YOLOv12 Large (Max Accuracy)",
            "description": "Maximum precision for static images."
        }
    }
    return jsonify(models_info)

@app.route("/api/detect-image", methods=["POST"])
def detect_image():
    """Detect faces in uploaded image."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({"error": "Invalid image format"}), 400

        conf_threshold = float(request.form.get("confidence", 0.5))
        blur_faces = request.form.get("blur_faces", "false").lower() == "true"
        model_key = request.form.get("model", "yolov12n")
        return_mode = request.form.get("return_mode", "base64")

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to read image"}), 400

        img_h, img_w = image.shape[:2]
        detector = get_detector(conf_threshold, model_key)
        detections = detector.detect_faces(image, conf_threshold=conf_threshold)

        if return_mode == "coords":
            return jsonify(
                {
                    "success": True,
                    "num_faces": len(detections),
                    "img_w": img_w,
                    "img_h": img_h,
                    "detections": [
                        {
                            "id": i + 1,
                            "x1": int(d["x1"]),
                            "y1": int(d["y1"]),
                            "x2": int(d["x2"]),
                            "y2": int(d["y2"]),
                            "w": int(d["w"]),
                            "h": int(d["h"]),
                            "confidence": round(float(d["confidence"]), 4),
                        }
                        for i, d in enumerate(detections)
                    ],
                }
            )

        output_image = detector.draw_faces(
            image, detections, blur_faces=blur_faces
        )
        face_crops = detector.get_face_crops(image, detections)

        _, buffer = cv2.imencode(
            ".jpg", output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        faces_data = []
        for i, crop_data in enumerate(face_crops):
            _, crop_buffer = cv2.imencode(
                ".jpg", crop_data["image"], [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            crop_base64 = base64.b64encode(crop_buffer).decode("utf-8")
            faces_data.append(
                {
                    "id": i + 1,
                    "image": f"data:image/jpeg;base64,{crop_base64}",
                    "confidence": round(crop_data["confidence"] * 100, 2),
                    "bbox": crop_data["bbox"],
                }
            )

        return jsonify(
            {
                "success": True,
                "num_faces": len(detections),
                "output_image": f"data:image/jpeg;base64,{img_base64}",
                "faces": faces_data,
                "detections": [
                    {
                        "id": i + 1,
                        "x1": int(d["x1"]),
                        "y1": int(d["y1"]),
                        "w": int(d["w"]),
                        "h": int(d["h"]),
                        "confidence": round(float(d["confidence"]) * 100, 2),
                    }
                    for i, d in enumerate(detections)
                ],
            }
        )

    except Exception as e:
        logger.error(f"Error in detect_image: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _process_video_job(
    job_id: str,
    input_path: str,
    output_path: str,
    output_filename: str,
    conf_threshold: float,
    blur_faces: bool,
    model_key: str,
):
    """Background worker function to process video and update progress."""
    try:
        detector = get_detector(conf_threshold, model_key)
        job_start_time = time.time()

        def update_progress(processed: int, total: int):
            pct = int(round((processed / total) * 100)) if total > 0 else 0
            pct = min(99, max(0, pct))
            elapsed = max(0.1, time.time() - job_start_time)
            fps_val = round(processed / elapsed, 1)
            remaining = total - processed
            eta_val = int(remaining / fps_val) if fps_val > 0 else 0
            
            with video_jobs_lock:
                if job_id in video_jobs:
                    video_jobs[job_id]["processed_frames"] = processed
                    video_jobs[job_id]["total_frames"] = total
                    video_jobs[job_id]["progress"] = pct
                    video_jobs[job_id]["fps"] = fps_val
                    video_jobs[job_id]["eta_seconds"] = eta_val

        stats = detector.detect_from_video(
            video_path=input_path,
            output_path=output_path,
            show_preview=False,
            blur_faces=blur_faces,
            progress_callback=update_progress,
        )

        with video_jobs_lock:
            if job_id in video_jobs:
                video_jobs[job_id].update(
                    {
                        "status": "completed",
                        "progress": 100,
                        "processed_frames": stats["total_frames"],
                        "total_frames": stats["total_frames"],
                        "stats": stats,
                        "video_url": f"/outputs/{output_filename}",
                        "download_url": f"/outputs/{output_filename}?download=true",
                    }
                )
    except Exception as e:
        logger.error(f"Video job {job_id} failed: {e}", exc_info=True)
        with video_jobs_lock:
            if job_id in video_jobs:
                video_jobs[job_id].update(
                    {"status": "failed", "error": str(e)}
                )
    finally:
        try:
            Path(input_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/detect-video", methods=["POST"])
def detect_video():
    """Upload a video and start asynchronous face detection job."""
    try:
        cleanup_old_files()

        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({"error": "Invalid video format"}), 400

        conf_threshold = float(request.form.get("confidence", 0.5))
        blur_faces = request.form.get("blur_faces", "false").lower() == "true"
        model_key = request.form.get("model", "yolov12n")

        job_id = uuid.uuid4().hex
        safe_name = secure_filename(file.filename)
        input_filename = f"{job_id}_{safe_name}"
        output_filename = f"detected_{job_id}.mp4"

        input_path = str(UPLOAD_FOLDER / input_filename)
        output_path = str(OUTPUT_FOLDER / output_filename)

        file.save(input_path)

        with video_jobs_lock:
            video_jobs[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "progress": 0,
                "processed_frames": 0,
                "total_frames": 0,
                "fps": 0.0,          
                "eta_seconds": 0,    
                "created_at": time.time(),
            }

        worker = threading.Thread(
            target=_process_video_job,
            args=(
                job_id,
                input_path,
                output_path,
                output_filename,
                conf_threshold,
                blur_faces,
                model_key,
            ),
            daemon=True,
        )
        worker.start()

        return jsonify(
            {"success": True, "job_id": job_id, "status": "processing"}
        )

    except Exception as e:
        logger.error(f"Error in detect_video: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/video-status/<job_id>", methods=["GET"])
def get_video_status(job_id: str):
    """Poll video processing status (kept for fallback & backward compatibility)."""
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(dict(job))


@app.route("/api/video-stream/<job_id>", methods=["GET"])
def stream_video_status(job_id: str):
    """
    Server-Sent Events (SSE) endpoint for real-time video progress.
    Replaces repeated HTTP polling with a single persistent event stream.
    """
    def event_generator():
        last_processed = -1
        last_status = None

        while True:
            with video_jobs_lock:
                job = video_jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                snapshot = dict(job)

            curr_processed = snapshot.get("processed_frames", 0)
            curr_status = snapshot.get("status")

            if curr_processed != last_processed or curr_status != last_status:
                last_processed = curr_processed
                last_status = curr_status
                yield f"data: {json.dumps(snapshot)}\n\n"

            if curr_status in ("completed", "failed"):
                break

            time.sleep(0.35)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers=headers,
    )


@app.route("/api/webcam-frame", methods=["POST"])
def detect_webcam_frame():
    """Detect faces in a single webcam frame sent from browser."""
    try:
        conf_threshold = float(request.form.get("confidence", 0.32))
        model_key = request.form.get("model", "yolov12n")
        frame = None

        image_file = request.files.get("file") or request.files.get("frame")
        if image_file:
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            conf_threshold = float(data.get("confidence", 0.32))
            model_key = data.get("model", "yolov12n")
            frame_b64 = data.get("frame", "")
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            if frame_b64:
                img_bytes = base64.b64decode(frame_b64)
                frame = cv2.imdecode(
                    np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                )

        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400

        img_h, img_w = frame.shape[:2]
        detector = get_detector(conf_threshold, model_key)
        detections = detector.detect_faces(frame, conf_threshold=conf_threshold)

        return jsonify(
            {
                "success": True,
                "frame_width": img_w,
                "frame_height": img_h,
                "num_faces": len(detections),
                "detections": {
                    "count": len(detections),
                    "boxes": [
                        {
                            "x1": int(d["x1"]),
                            "y1": int(d["y1"]),
                            "x2": int(d["x2"]),
                            "y2": int(d["y2"]),
                            "w": int(d["w"]),
                            "h": int(d["h"]),
                            "conf": float(d["confidence"]),
                        }
                        for d in detections
                    ],
                },
            }
        )

    except Exception as e:
        logger.error(f"Error in detect_webcam_frame: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    """Serve processed video files with optional download attachment."""
    as_attachment = request.args.get("download", "false").lower() == "true"
    return send_from_directory(
        app.config["OUTPUT_FOLDER"], filename, as_attachment=as_attachment
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for Docker / Cloud deployment."""
    return jsonify(
        {
            "status": "healthy",
            "loaded_models": list(detectors.keys()),
            "max_loaded_models": MAX_LOADED_MODELS,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
