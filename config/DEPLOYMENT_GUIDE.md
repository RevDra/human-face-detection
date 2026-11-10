# 📋 YOLOv12 Face Detection - Project Overview & Guide (v2.0.0)

## 🎯 Quick Reference

### 🚀 Start Here

1. Read: **DEPLOYMENT_GUIDE.md** (5 min)
2. Run (Linux/Mac): `bash deploy.sh start`
3. Run (Windows): `deploy.bat start`
4. Visit: http://localhost:7860

---

## 📁 Complete File List

### Core Application Files (Required)

* `src/web_app.py`: Main FastAPI server (Port 7860), optimized for async ASGI and real-time SSE.
* `src/face_detector.py`: Optimized YOLOv12 logic core (OpenCV + ONNX Runtime + NumPy vectorized interpolation).
* `web/templates/index.html`: Responsive UI with adaptive webcam pacing and in-place smoothing.

### Deployment Files

* `config/Dockerfile`: Docker configuration (Python 3.11-slim + Uvicorn ASGI).
* `config/docker-compose.yml`: Container orchestration.
* `config/deploy.sh`: Automation script for Linux/Mac.
* `config/deploy.bat`: Automation script for Windows.
* `.github/workflows/docker-publish.yml`: CI/CD for auto-building Docker images.
* `.github/workflows/release.yml`: Automated Semantic Release CI/CD.

### Models (YOLOv12 INT8 ONNX)

* `models/yolov12n-face.onnx`: Nano (Best for Webcam/CPU).
* `models/yolov12s-face.onnx`: Small (Balanced).
* `models/yolov12m-face.onnx`: Medium (Best for Video).
* `models/yolov12l-face.onnx`: Large (Best for Static Images).

### Documentation Files

✅ DEPLOYMENT_GUIDE.md (THIS FILE)

* Overview of all files created
* Quick reference guide
* File descriptions
* Recommended reading order

### Configuration Files

```
- requirements.txt
   + fastapi
   + uvicorn[standard]
   + pydantic
   + python-multipart
   + onnxruntime
   + opencv-python-headless
   + All other dependencies intact

```

### Directory Structure

```
Human_face_detection/
├── 🤖 CI/CD & AUTOMATION
│   └── .github/
│       └── workflows/
│           ├── docker-publish.yml        ⭐ GitHub Action to Auto-build Docker Image
│           ├── release.yml               ⭐ Automated Semantic Release CI/CD
│           └── lint.yml                  ⭐ Code Quality Check
│
├── ⚙️ DEPLOYMENT & CONFIG
│   └── config/
│       ├── Dockerfile                    ⭐ Docker image config
│       ├── docker-compose.yml            ⭐ Docker Compose setup
│       ├── deploy.sh                     Linux/macOS deployment script
│       └── deploy.bat                    Windows deployment script
│
├── 🌐 SOURCE CODE
│   └── src/
│       ├── web_app.py                    ⭐ Main FastAPI server (Port 7860)
│       └── face_detector.py              ⭐ YOLOv12 Detection Engine (ONNX + OpenCV)
│
├── 🎨 FRONTEND
│   └── web/templates/
│       └── index.html                    ⭐ Web UI (Adaptive Webcam Pacing)
│
├── 🧠 MODELS (YOLOv12 INT8 ONNX)
│   └── models/
│       ├── yolov12n-face.onnx       Nano model (Best for Webcam)
│       ├── yolov12s-face.onnx       Small model
│       ├── yolov12m-face.onnx       Medium model
│       └── yolov12l-face.onnx       Large model
│
├── 📚 DOCUMENTATION & META
│   ├── README.md                         Main documentation
│   ├── LICENSE                           AGPL v3 License
│   ├── SECURITY.md                       Security policy
│   ├── requirements.txt                  Python dependencies
│   └── DEPLOYMENT_GUIDE.md               This file
│
└── 📁 RUNTIME STORAGE (Auto-created)
    ├── uploads/                          Temporary file storage
    └── outputs/                          Processed video outputs

```

---

## 🎯 What Each File Does

### src/web_app.py (Main Application)

```python
# Key features:
- FastAPI ASGI web server
- REST API endpoints & SSE Real-time Streaming
  - POST /api/detect-image
  - POST /api/detect-video
  - GET /api/video-stream/{job_id}
  - GET /api/models
  - GET /api/health
- ThreadPoolExecutor & BackgroundTasks
- O(1) memory chunked file uploads
- Robust path traversal safety guards
- Size: ~400 lines
- Depends on: face_detector.py, FastAPI, Uvicorn

```

### web/templates/index.html (Web Interface)

```html
<!-- Features: -->
- Beautiful gradient background
- Responsive design & Mobile-friendly
- Adaptive webcam pacing & in-place smoothing
- Real-time SSE progress indicators for video processing
- Drag-and-drop file upload
- JavaScript API integration
- Size: ~800 lines

```

### docker-compose.yml

```yaml
# Includes:
- Service definition
- Port mapping (7860:7860)
- Volume mounting
- Health checks
- Restart policy

```

### deploy.sh & deploy.bat (Deployment Automation)

```bash
# Features:
- Virtual environment setup
- Dependency installation
- Uvicorn ASGI server integration
- Port management & auto-launch

```

---

## 🚀 Deployment Options

### Option 1: Local Development (Easiest)

```bash
uvicorn src.web_app:app --host 0.0.0.0 --port 7860 --reload
# Visit http://localhost:7860

```

**Time:** 2 minutes
**Cost:** Free
**Best for:** Testing, development

### Option 2: Docker Local

```bash
docker-compose -f config/docker-compose.yml up --build

```

**Time:** 5 minutes
**Cost:** Free
**Best for:** Consistent environment

### Option 3: Hugging Face Spaces (Recommended)

1. Create new Space (Docker SDK).
2. Upload all files.
3. Ensure `Dockerfile` is configured at root or path.
4. App runs automatically on port **7860**.

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | /api/detect-image | Upload image & get detection result (Base64/JSON) |
| POST | /api/detect-video | Upload video in O(1) chunks & start async job |
| GET | /api/video-stream/{job_id} | Real-time Server-Sent Events (SSE) progress stream |
| GET | /api/models | Get list of available YOLOv12 models |
| GET | /api/health | Server status check |
| GET | /outputs/{filename} | Download or stream processed video files (Range Requests supported) |

---

## 📊 Performance & Architecture Improvements (v2.0.0)

* **Asynchronous ASGI Performance:** Migrated from synchronous WSGI Flask to non-blocking FastAPI and Uvicorn.
* **Vectorized Interpolation:** Optimized `face_detector.py` using NumPy vectorization and motion-gated logic.
* **O(1) Memory Footprint:** Chunked file uploads (1MB blocks) prevent server memory overload on large videos.
* **Adaptive Frontend Pacing:** Replaced rigid intervals with dynamic timeout scheduling and smooth object mutation.

---

**Created:** January 31, 2026

**Updated:** June 2026 (v2.0.0 FastAPI Architecture Release)

**Status:** Production Ready ✅
