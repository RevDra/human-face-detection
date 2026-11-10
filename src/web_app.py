"""
FastAPI Web Application for YOLOv12 Human Face Detection.
Provides REST APIs and Web UI for image, video, and webcam detection
with 100% backward compatibility with the existing frontend.
"""

import asyncio
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
from typing import Any, AsyncIterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from werkzeug.utils import secure_filename

from .face_detector import YOLOv12FaceDetector

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

MAX_VIDEO_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_WEBCAM_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_PIXELS = 8192 * 8192
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB chunked write (O(1) RAM)

MAX_VIDEO_WORKERS = int(os.environ.get("MAX_VIDEO_WORKERS", 2))
MAX_INFERENCE_WORKERS = int(
    os.environ.get("MAX_INFERENCE_WORKERS", max(2, (os.cpu_count() or 4) // 2))
)

video_executor = ThreadPoolExecutor(max_workers=MAX_VIDEO_WORKERS)
inference_executor = ThreadPoolExecutor(max_workers=MAX_INFERENCE_WORKERS)

_last_cleanup_ts: float = 0.0
_cleanup_lock = threading.Lock()

app = FastAPI(
    title="YOLOv12 Human Face Detection API",
    version="1.9.1",
)

if STATIC_FOLDER.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_FOLDER))

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


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Ensure all HTTP errors use {'error': ...} format for frontend compatibility."""
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert FastAPI 422 validation errors into 400 {'error': ...} responses."""
    return JSONResponse({"error": "Invalid request parameters"}, status_code=400)


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


def _decode_image_bytes(data: bytes, max_bytes: int = MAX_IMAGE_SIZE_BYTES) -> np.ndarray | None:
    """
    Safely decode raw image bytes into a BGR image with size and pixel-count protection.

    Args:
        data: Raw image bytes.
        max_bytes: Maximum allowed byte length.

    Returns:
        Decoded BGR image as np.ndarray, or None if invalid.
    """
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
        min_interval: Minimum seconds between consecutive filesystem scans.

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


def _parse_bool(raw_val: Any) -> bool:
    """Parse boolean form/JSON field safely."""
    if isinstance(raw_val, bool):
        return raw_val
    return str(raw_val).strip().lower() in ("true", "1", "yes", "on")


def get_detector(conf_threshold: float = 0.32, model_key: str = "yolov12n") -> YOLOv12FaceDetector:
    """
    Get or initialize a YOLOv12FaceDetector instance with LRU memory eviction.

    Args:
        conf_threshold: Default confidence threshold when initializing a new model.
        model_key: Model identifier key in AVAILABLE_MODELS.

    Returns:
        Cached or newly loaded YOLOv12FaceDetector instance.
    """
    if model_key not in AVAILABLE_MODELS:
        model_key = "yolov12n"

    with detectors_lock:
        if model_key in detectors:
            detectors.move_to_end(model_key)
            return detectors[model_key]

        while len(detectors) >= MAX_LOADED_MODELS:
            evicted_key, _ = detectors.popitem(last=False)
            logger.info("Evicted model '%s' from LRU cache to free RAM.", evicted_key)
            gc.collect()

        model_path = AVAILABLE_MODELS[model_key]
        logger.info("Loading model '%s' from %s...", model_key, model_path)
        detector = YOLOv12FaceDetector(model_path=model_path, conf_threshold=conf_threshold)
        detectors[model_key] = detector
        return detector


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render main web interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/models")
async def get_models() -> JSONResponse:
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
    return JSONResponse(models_info)


def _run_image_detection_sync(
    raw_bytes: bytes,
    conf_threshold: float,
    blur_faces: bool,
    model_key: str,
    return_mode: str,
) -> tuple[dict[str, Any], int]:
    """Synchronous worker for CPU-bound image decoding and ONNX inference."""
    image = _decode_image_bytes(raw_bytes, max_bytes=MAX_IMAGE_SIZE_BYTES)
    if image is None:
        return {"error": "Failed to read image or image exceeds limit"}, 400

    img_h, img_w = image.shape[:2]
    detector = get_detector(conf_threshold, model_key)
    detections = detector.detect_faces(image, conf_threshold=conf_threshold)

    if return_mode == "coords":
        return (
            {
                "success": True,
                "num_faces": len(detections),
                "img_w": img_w,
                "img_h": img_h,
                "detections": _format_detections(detections, scale_percent=False, include_xy2=True),
            },
            200,
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

    return (
        {
            "success": True,
            "num_faces": len(detections),
            "output_image": f"data:image/jpeg;base64,{_encode_jpeg_b64(output_image)}",
            "faces": faces_data,
            "detections": _format_detections(detections, scale_percent=True, include_xy2=False),
        },
        200,
    )


@app.post("/api/detect-image")
async def detect_image(
    image: UploadFile | None = File(default=None),
    confidence: str = Form(default="0.5"),
    blur_faces: str = Form(default="false"),
    model: str = Form(default="yolov12n"),
    return_mode: str = Form(default="base64"),
) -> JSONResponse:
    """Detect faces in an uploaded image and return coordinates or Base64 results."""
    try:
        if image is None or not image.filename:
            return JSONResponse({"error": "No image provided or empty filename"}, status_code=400)
        if not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
            return JSONResponse({"error": "Invalid image format"}, status_code=400)

        raw_bytes = await image.read(MAX_IMAGE_SIZE_BYTES + 1)
        if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
            return JSONResponse({"error": "Image file too large"}, status_code=400)

        conf_val = _parse_confidence(confidence, default=0.5)
        blur_flag = _parse_bool(blur_faces)

        loop = asyncio.get_running_loop()
        payload, status_code = await loop.run_in_executor(
            inference_executor,
            _run_image_detection_sync,
            raw_bytes,
            conf_val,
            blur_flag,
            model,
            return_mode,
        )
        return JSONResponse(payload, status_code=status_code)
    except Exception as exc:
        logger.error("Error in detect_image: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Internal server error during image detection"}, status_code=500
        )


def _process_video_job(
    job_id: str,
    input_path: str,
    output_path: str,
    output_filename: str,
    conf_threshold: float,
    blur_faces: bool,
    model_key: str,
) -> None:
    """Background worker function to process video and update progress."""
    try:
        detector = get_detector(conf_threshold, model_key)
        job_start_time = time.time()

        def update_progress(processed: int, total: int) -> None:
            pct = int(round((processed / total) * 100)) if total > 0 else 0
            pct = min(99, max(0, pct))
            elapsed = max(0.1, time.time() - job_start_time)
            fps_val = round(processed / elapsed, 1)
            remaining = total - processed
            eta_val = int(remaining / fps_val) if fps_val > 0 else 0

            with video_jobs_lock:
                if job_id in video_jobs:
                    video_jobs[job_id].update(
                        {
                            "processed_frames": processed,
                            "total_frames": total,
                            "progress": pct,
                            "fps": fps_val,
                            "eta_seconds": eta_val,
                        }
                    )

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
        logger.error("Error in background video job %s: %s", job_id, exc, exc_info=True)
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
        except OSError:
            pass


@app.post("/api/detect-video")
async def detect_video(
    video: UploadFile | None = File(default=None),
    confidence: str = Form(default="0.5"),
    blur_faces: str = Form(default="false"),
    model: str = Form(default="yolov12n"),
) -> JSONResponse:
    """Upload a video in O(1) memory chunks and start asynchronous face detection job."""
    input_path_obj: Path | None = None
    try:
        cleanup_old_files()

        if video is None or not video.filename:
            return JSONResponse({"error": "No video file provided"}, status_code=400)
        if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
            return JSONResponse({"error": "Invalid video format"}, status_code=400)

        conf_threshold = _parse_confidence(confidence, default=0.5)
        blur_flag = _parse_bool(blur_faces)

        job_id = uuid.uuid4().hex
        safe_name = secure_filename(video.filename) or "video.mp4"
        input_filename = f"{job_id}_{safe_name}"
        output_filename = f"detected_{job_id}.mp4"

        input_path_obj = UPLOAD_FOLDER / input_filename
        output_path = str(OUTPUT_FOLDER / output_filename)

        written_bytes = 0
        with open(input_path_obj, "wb") as out_file:
            while True:
                chunk = await video.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                written_bytes += len(chunk)
                if written_bytes > MAX_VIDEO_SIZE_BYTES:
                    out_file.close()
                    input_path_obj.unlink(missing_ok=True)
                    return JSONResponse({"error": "Video file exceeds 1GB limit"}, status_code=400)
                out_file.write(chunk)

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
            str(input_path_obj),
            output_path,
            output_filename,
            conf_threshold,
            blur_flag,
            model,
        )

        return JSONResponse({"success": True, "job_id": job_id, "status": "processing"})
    except Exception as exc:
        if input_path_obj is not None:
            input_path_obj.unlink(missing_ok=True)
        logger.error("Error in detect_video: %s", exc, exc_info=True)
        return JSONResponse({"error": "Internal server error during video upload"}, status_code=500)


@app.get("/api/video-status/{job_id}")
async def get_video_status(job_id: str) -> JSONResponse:
    """Poll video processing status (kept for fallback & backward compatibility)."""
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if not job:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        return JSONResponse(dict(job))


@app.get("/api/video-stream/{job_id}")
async def stream_video_status(job_id: str) -> StreamingResponse:
    """
    Non-blocking Server-Sent Events (SSE) endpoint for real-time video progress.
    Uses asyncio.sleep so 0 OS threads are blocked while streaming.
    """

    async def event_generator() -> AsyncIterator[str]:
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

            await asyncio.sleep(0.3)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


def _run_webcam_detection_sync(
    raw_bytes: bytes, conf_threshold: float, model_key: str
) -> tuple[dict[str, Any], int]:
    """Synchronous worker for decoding and detecting faces in a webcam frame."""
    frame = _decode_image_bytes(raw_bytes, max_bytes=MAX_WEBCAM_SIZE_BYTES)
    if frame is None:
        return {"error": "Invalid frame data"}, 400

    img_h, img_w = frame.shape[:2]
    detector = get_detector(conf_threshold, model_key)
    detections = detector.detect_faces(frame, conf_threshold=conf_threshold)

    return (
        {
            "success": True,
            "frame_width": img_w,
            "frame_height": img_h,
            "num_faces": len(detections),
            "detections": {
                "count": len(detections),
                "boxes": [
                    {
                        "x1": d["x1"],
                        "y1": d["y1"],
                        "x2": d["x2"],
                        "y2": d["y2"],
                        "w": d["w"],
                        "h": d["h"],
                        "conf": float(d["confidence"]),
                    }
                    for d in detections
                ],
            },
        },
        200,
    )


@app.post("/api/webcam-frame")
async def detect_webcam_frame(request: Request) -> JSONResponse:
    """
    Detect faces in a single webcam frame sent from browser.
    Supports both multipart/form-data (Blob) and application/json (Base64).
    """
    try:
        content_type = request.headers.get("content-type", "")
        conf_threshold = 0.32
        model_key = "yolov12n"
        raw_bytes: bytes = b""

        if "application/json" in content_type:
            data = await request.json()
            conf_threshold = _parse_confidence(data.get("confidence", 0.32), 0.32)
            model_key = str(data.get("model", "yolov12n"))
            frame_b64 = str(data.get("frame", ""))
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            if frame_b64:
                raw_bytes = base64.b64decode(frame_b64)
        else:
            form = await request.form()
            conf_threshold = _parse_confidence(form.get("confidence", 0.32), 0.32)
            model_key = str(form.get("model", "yolov12n"))
            upload_obj = form.get("file") or form.get("frame")
            if upload_obj is not None and hasattr(upload_obj, "read"):
                raw_bytes = await upload_obj.read(MAX_WEBCAM_SIZE_BYTES + 1)

        if not raw_bytes or len(raw_bytes) > MAX_WEBCAM_SIZE_BYTES:
            logger.warning("Webcam frame rejected: empty or exceeds size limit.")
            return JSONResponse({"error": "Invalid or empty frame data"}, status_code=400)

        loop = asyncio.get_running_loop()
        payload, status_code = await loop.run_in_executor(
            inference_executor,
            _run_webcam_detection_sync,
            raw_bytes,
            conf_threshold,
            model_key,
        )
        return JSONResponse(payload, status_code=status_code)
    except Exception as exc:
        logger.error("Error in detect_webcam_frame: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Internal server error during webcam detection"}, status_code=500
        )


@app.get("/outputs/{filename:path}")
async def serve_output(filename: str, download: str = "false") -> Response:
    """Serve processed video files with HTTP Range support and optional download attachment."""
    # 1. Chặn tuyệt đối các ký tự điều hướng thư mục nguy hiểm hoặc tên file không hợp lệ
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", "..") or "/" in filename or "\\" in filename:
        return JSONResponse({"error": "Invalid file path"}, status_code=400)

    safe_root = OUTPUT_FOLDER.resolve()
    target_path = (OUTPUT_FOLDER / safe_name).resolve()

    try:
        target_path.relative_to(safe_root)
    except ValueError:
        return JSONResponse({"error": "File not found"}, status_code=404)

    if not target_path.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)

    as_attachment = download.lower() in ("true", "1", "yes")
    disposition = "attachment" if as_attachment else "inline"
    return FileResponse(
        path=str(target_path),
        media_type="video/mp4",
        filename=target_path.name if as_attachment else None,
        content_disposition_type=disposition,
    )


@app.post("/api/feedback")
async def submit_feedback(request: Request) -> JSONResponse:
    """Receive user rating and feedback from Web UI."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    logger.info("User feedback received: %s", data)
    return JSONResponse({"success": True})


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for Docker / Cloud deployment."""
    return JSONResponse(
        {
            "status": "healthy",
            "loaded_models": list(detectors.keys()),
            "max_loaded_models": MAX_LOADED_MODELS,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=debug)
