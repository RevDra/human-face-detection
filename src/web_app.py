"""
Flask Web Application for YOLOv12 Human Face Detection
Provides REST APIs and Web UI for image, video, and webcam detection
"""

import base64
import gc
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

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
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from face_detector import YOLOv12FaceDetector

MAX_VIDEO_WORKERS = int(os.environ.get("MAX_VIDEO_WORKERS", 2))
video_executor = ThreadPoolExecutor(max_workers=MAX_VIDEO_WORKERS)
_last_cleanup_ts: float = 0.0
_cleanup_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_IMAGE_PIXELS = 8192 * 8192

AVAILABLE_MODELS = {
    "yolov12n": "models/yolov12n-face.onnx",
    "yolov12s": "models/yolov12s-face.onnx",
    "yolov12m": "models/yolov12m-face.onnx",
    "yolov12l": "models/yolov12l-face.onnx",
}

MAX_LOADED_MODELS = int(os.environ.get("MAX_LOADED_MODELS", 2))
detectors: "OrderedDict[str, YOLOv12FaceDetector]" = OrderedDict()
detectors_lock = threading.Lock()

video_jobs: dict[str, dict[str, Any]] = {}
video_jobs_lock = threading.Lock()


def _encode_jpeg_b64(image: np.ndarray, quality: int = 85) -> str:
    """
    Encode a BGR NumPy image into a Base64 JPEG string.

    Args:
        image: Input BGR image array.
        quality: JPEG compression quality (0 to 100).

    Returns:
        Base64-encoded JPEG string.
    """
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


def _decode_image_storage(
    file_storage: FileStorage, max_bytes: int = MAX_IMAGE_SIZE_BYTES
) -> np.ndarray | None:
    """
    Safely read and decode an image with byte-size and pixel-count protection.
    """
    data = file_storage.read(max_bytes + 1)
    if not data or len(data) > max_bytes:
        return None
    file_bytes = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if h * w > MAX_IMAGE_PIXELS:
        return None
    return image


def _format_detections(
    detections: list[dict[str, Any]],
    scale_percent: bool = False,
    include_xy2: bool = True,
) -> list[dict[str, Any]]:
    """
    Serialize raw detector outputs into standardized JSON-ready dictionaries.

    Args:
        detections: List of detection dicts from YOLOv12FaceDetector.
        scale_percent: Multiply confidence by 100 and round to 2 decimals if True.
        include_xy2: Include x2 and y2 keys in each item if True.

    Returns:
        Formatted list of detection dictionaries.
    """
    formatted: list[dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        conf = (
            round(float(det["confidence"]) * 100, 2)
            if scale_percent
            else round(float(det["confidence"]), 4)
        )
        item: dict[str, Any] = {
            "id": idx,
            "x1": det["x1"],
            "y1": det["y1"],
            "w": det["w"],
            "h": det["h"],
            "confidence": conf,
        }
        if include_xy2:
            item["x2"] = det["x2"]
            item["y2"] = det["y2"]
        formatted.append(item)
    return formatted


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    """
    Check if a filename has an allowed extension.

    Args:
        filename: Name of the uploaded file.
        allowed_extensions: Set of permitted lowercase file extensions.

    Returns:
        True if the file extension is valid, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def cleanup_old_files(max_age_seconds: int = 3600, min_interval: int = 300) -> None:
    """
    Remove temporary uploaded/output files and job records older than max_age_seconds.

    Args:
        max_age_seconds: Maximum lifetime in seconds for temporary artifacts.

    Returns:
        None
    """
    global _last_cleanup_ts
    now = time.time()
    with _cleanup_lock:
        if now - _last_cleanup_ts < min_interval:
            return
        _last_cleanup_ts = now
    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        try:
            for file_path in folder.iterdir():
                if (
                    file_path.is_file()
                    and file_path.name != ".gitkeep"
                    and now - file_path.stat().st_mtime > max_age_seconds
                ):
                    file_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Cleanup warning in %s: %s", folder, exc)

    with video_jobs_lock:
        expired_jobs = [
            jid
            for jid, job in video_jobs.items()
            if now - job.get("created_at", now) > max_age_seconds
        ]
        for jid in expired_jobs:
            video_jobs.pop(jid, None)


def _parse_confidence(raw_val: Any, default: float = 0.5) -> float:
    """
    Safely parse and clamp confidence threshold into [0.05, 0.99].

    Args:
        raw_val: Raw input value from form or JSON.
        default: Fallback confidence value if parsing fails.

    Returns:
        Clamped float confidence threshold.
    """
    try:
        val = float(raw_val)
        if not math.isfinite(val):
            return default
        return max(0.05, min(0.99, val))
    except (TypeError, ValueError):
        return default


def get_detector(conf_threshold: float = 0.32, model_key: str = "yolov12n") -> YOLOv12FaceDetector:
    """
    Get or initialize a YOLOv12FaceDetector instance with LRU memory eviction.
    """
    if model_key not in AVAILABLE_MODELS:
        model_key = "yolov12n"

    with detectors_lock:
        if model_key in detectors:
            detectors.move_to_end(model_key)
            detector = detectors[model_key]
            return detector

        while len(detectors) >= MAX_LOADED_MODELS:
            evicted_key, _ = detectors.popitem(last=False)
            logger.info(f"Evicted model '{evicted_key}' from LRU cache to free RAM.")
            gc.collect()

        model_path = AVAILABLE_MODELS[model_key]
        logger.info(f"Loading model '{model_key}' from {model_path}...")
        detector = YOLOv12FaceDetector(model_path=model_path, conf_threshold=conf_threshold)
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
            "description": "Optimized for speed, ideal for webcam.",
        },
        "small": {
            "name": "yolov12s",
            "label": "YOLOv12 Small (Balanced)",
            "description": "Good balance between speed and accuracy.",
        },
        "medium": {
            "name": "yolov12m",
            "label": "YOLOv12 Medium (High Accuracy)",
            "description": "Higher precision for video analysis.",
        },
        "large": {
            "name": "yolov12l",
            "label": "YOLOv12 Large (Max Accuracy)",
            "description": "Maximum precision for static images.",
        },
    }
    return jsonify(models_info)


@app.route("/api/detect-image", methods=["POST"])
def detect_image() -> tuple[Response, int] | Response:
    """
    Detect faces in an uploaded image and return coordinates or Base64 results.

    Returns:
        Flask JSON Response containing face detections.
    """
    try:
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify({"error": "No image provided or empty filename"}), 400
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({"error": "Invalid image format"}), 400

        conf_threshold = float(request.form.get("confidence", 0.5))
        blur_faces = request.form.get("blur_faces", "false").lower() == "true"
        model_key = request.form.get("model", "yolov12n")
        return_mode = request.form.get("return_mode", "base64")

        image = _decode_image_storage(file)
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
                    "detections": _format_detections(
                        detections, scale_percent=False, include_xy2=True
                    ),
                }
            )

        output_image = detector.draw_faces(image, detections, blur_faces=blur_faces)
        face_crops = detector.get_face_crops(image, detections)

        faces_data = [
            {
                "id": i,
                "image": f"data:image/jpeg;base64,{_encode_jpeg_b64(crop['image'])}",
                "confidence": round(crop["confidence"] * 100, 2),
                "bbox": crop["bbox"],
            }
            for i, crop in enumerate(face_crops, start=1)
        ]

        return jsonify(
            {
                "success": True,
                "num_faces": len(detections),
                "output_image": f"data:image/jpeg;base64,{_encode_jpeg_b64(output_image)}",
                "faces": faces_data,
                "detections": _format_detections(detections, scale_percent=True, include_xy2=False),
            }
        )
    except Exception as exc:
        logger.error("Error in detect_image: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error during image detection"}), 500


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
            conf_threshold=conf_threshold,
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
    except Exception as exc:
        app.logger.exception(f"Error in background video job {job_id}")
        with video_jobs_lock:
            if job_id in video_jobs:
                video_jobs[job_id].update(
                    {
                        "status": "failed",
                        "error": "Failed to process video",
                    }
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

        video_executor.submit(
            _process_video_job,
            job_id,
            input_path,
            output_path,
            output_filename,
            conf_threshold,
            blur_faces,
            model_key,
        )

        return jsonify({"success": True, "job_id": job_id, "status": "processing"})

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
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

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
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=as_attachment)


@app.route("/api/feedback", methods=["POST"])
def submit_feedback() -> tuple[Response, int] | Response:
    """Receive user rating and feedback from Web UI."""
    data = request.get_json(silent=True) or {}
    logger.info("User feedback received: %s", data)
    return jsonify({"success": True})


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
