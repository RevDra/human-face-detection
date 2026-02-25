# 🔍 YOLOv12 Human Face Detection Web Application

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/RevDra/YOLOv12_HFD)
[![License: AGPL v3](https://img.shields.io/badge/%E2%9A%96%EF%B8%8F_License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-2496ED?logo=docker&logoColor=white)](config/Dockerfile)
[![Build Status](https://github.com/RevDra/human-face-detection/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/RevDra/human-face-detection/actions/workflows/docker-publish.yml)

<br>
<div align="center">
  <img src="assets/demo_ui.png" alt="YOLOv12 Face Detection Interface" width="100%">
</div>
<br>

A professional, real-time face detection system built with YOLOv12 and Flask. This project leverages the latest Attention Mechanism features of YOLOv12 to detect faces in images, videos, and live webcam streams with state-of-the-art accuracy and speed. 

View the demo using this [link](https://revdra-yolov12-hfd.hf.space/).

## ✨ Features

### 📷 Image & Video Detection
- Upload and detect faces in images (JPG, PNG) and videos (MP4, AVI, MOV).
- Attention-based detection for small, distant, or occluded faces.
- Interactive bounding boxes with cropped face previews.
- Download annotated results and face statistics.

### 📹 Live Webcam
- Real-time detection directly from your browser.
- Side-by-side video feed and detection results.
- Live statistics (FPS, face count, duration).

### 📊 Feedback & Analytics
- Built-in MySQL database integration (via Aiven) to collect user ratings and feedback securely.
- Ready for admin dashboard visualization.

### 🤖 Model Selection
- **YOLOv12 Nano** (yolov12n-face.pt) - Super Fast, best for CPU/Webcam.
- **YOLOv12 Small** (yolov12s-face.pt) - Balanced speed & accuracy.
- **YOLOv12 Medium** (yolov12m-face.pt) - High precision (The "Sweet Spot").
- **YOLOv12 Large** (yolov12l-face.pt) - State-of-the-art accuracy for high-res images.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/RevDra/human-face-detection.git](https://github.com/RevDra/human-face-detection.git)
cd human-face-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Setup (Required):**
Copy the template environment file and add your secure credentials.
```bash
cp .env.example .env
```
*Open `.env` in a text editor and update the `DB_URL` with your MySQL connection string.*

4. **Run the web server:**
#### Linux/Mac
```bash
./config/deploy.sh start
```

#### Windows
```bash
config\deploy.bat start
```

5. **Open in browser:** Navigate to `https://localhost:7860`

## 📁 Project Structure

```
Human_face_detection/
├── .github/                            # CI/CD & Automation
│   ├── ISSUE_TEMPLATE/                 # Community Forms
│   │   ├── bug_report.md               # Bug report template
│   │   ├── config.yml                  # Discussions link config
│   │   └── feature_request.md          # Feature request template
│   ├── workflows/
│   │   ├── docker-publish.yml          # Auto-build Docker Image
│   │   └── lint.yml                    # Quality Check (Black + Flake8 + isort + Mypy)
│   ├── dependabot.yml                  # Automated Dependency Updates
│   └── FUNDING.yml                     # Sponsor settings
│
├── assets/                             # Project Images & Screenshots
│   └── demo_ui.png                     # Interface preview for README
│
├── config/                             # Configuration & Deployment scripts
│   ├── Dockerfile                      # Docker image config
│   ├── docker-compose.yml              # Docker Compose setup
│   ├── deploy.sh                       # Linux deployment script
│   ├── deploy.bat                      # Windows deployment script
│   └── DEPLOYMENT_GUIDE.md             # Detailed deployment guide
│
├── models/                             # YOLOv12 Models
│   ├── yolov12n-face.pt                # Nano model (Fastest)
│   ├── yolov12s-face.pt                # Small model (Balanced)
│   ├── MODELS.md                       # Download instructions for Med/Large models
│   └── training/                       # Source code & benchmarks (Loss, PR curves, etc.)
│
├── src/                                # Source Code
│   ├── web_app.py                      # Flask web server
│   └── face_detection_yolov12.py       # YOLOv12 detection engine
│
├── web/                                # Frontend Assets
│   └── templates/
│       └── index.html                  # Web UI
│
├── .flake8                             # Flake8 Configuration
├── .dockerignore                       # Docker Ignore
├── .gitignore                          # Git Ignore
├── .gitattributes                      # Normalized code (LF)
├── pyproject.toml                      # Black + Mypy + isort Configuration
├── CODE_OF_CONDUCT.md                  # Community guidelines
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # AGPL v3 License
├── README.md                           # Main documentation
├── SECURITY.md                         # Security policy
├── .env.example                        # Template for environment variables (DB credentials)
└── requirements.txt                    # Python dependencies
```

## 🐳 Docker Support (Recommended)
You can run the application instantly without needing to install Python or manually install dependencies.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop) installed.

1. **Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)RevDra/human-face-detection.git
cd Human_face_detection
```
   
2. **Run with Docker Compose:**
```bash
# Build and run with Docker Compose (from project root)
docker-compose -f config/docker-compose.yml up --build

# Or build manually
docker build -t yolov12-face-detection -f config/Dockerfile .
docker run -p 7860:7860 -v $(pwd)/data:/app/data yolov12-face-detection
```
   
3. **Access the App:**
   Open `http://localhost:7860` in your browser.

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/api/detect-image` | Detect faces in uploaded image |
| POST | `/api/detect-video` | Detect faces in uploaded video |
| GET | `/api/models` | List available models |
| POST | `/api/feedback` | Submit user rating and comments to database |
| GET | `/api/health` | Health check |
| GET | `/api/download/<filename>` | Download processed files |

## 💻 Usage

### Via Web Interface
1. Select a detection model
2. Upload an image/video or start webcam
3. Wait for processing
4. View results and download if needed

### Via API (Python Example)
```python
import requests

# Detect faces in image
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'model': 'yolov12l-face.pt'}
    response = requests.post('http://localhost:7860/api/detect-image', 
                            files=files, data=data)
    result = response.json()
    
print(f"Detected {result['detections']['count']} faces")
```

## 📊 Detection Details

### Confidence Threshold
Default: 0.32 (32%)
- Higher threshold = fewer false positives but may miss faces
- Lower threshold = more detections but more false positives

### Output Includes
- Bounding box coordinates (x1, y1, x2, y2)
- Confidence score (0-100%)
- Face dimensions (width × height)
- Face position on image

## ⚙️ Configuration

Edit `web_app.py` to modify:
- `MAX_FILE_SIZE` - Maximum upload size (default: 500MB)
- `UPLOAD_FOLDER` - Temporary file location
- `PORT` - Application port (default: 7860)

## 📝 Notes

### Model Files Required
Three model files are required in the `models/` directory:
- `yolov12n-face.pt` (5.3 MB) ✅ Included
- `yolov12s-face.pt` (18.5 MB) ✅ Included
- `yolov12m_face.pt` (39.8 MB) 📥 [Download](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12m-face.pt)
- `yolov12l_face.pt` (52.3 MB) 📥 [Download](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt)

**See [models/MODELS.md](models/MODELS.md) for detailed download instructions.**

### Performance Tips
1. Use YOLOv12 Nano for webcam to achieve high FPS.
2. Use YOLOv12 Large for high-resolution static images.
3. If running on Hugging Face Spaces (CPU), stick to Nano or Small models.

### Browser Compatibility
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- IE: ❌ Not supported

## 🐛 Troubleshooting

### Models Not Found
```
FileNotFoundError: Model not found: models/yolov12m-face.pt
```
**Solution:** Download missing models from [models/MODELS.md](models/MODELS.md). Only `yolov12n-face.pt` and `yolov12s-face.pt` are included by default.

### Camera Permission Denied
**Solution:** Grant camera permission in browser settings. If deploying on a remote server, you must use HTTPS for the webcam to work.

### Out of Memory
**Solution:** Use a smaller model (Nano) or reduce video resolution

### Slow Detection
**Solution:** 
- Use YOLOv12 Nano
- Reduce input resolution
- Check CPU/GPU usage

## 📚 References

- [Model Setup Guide](models/MODELS.md) - Download and setup instructions
- [YOLOv12-Face Repository](https://github.com/YapaLab/yolo-face) - Source of the models/weights.
- [Ultralytics YOLOv12](https://docs.ultralytics.com/) - YOLOv12 documentation.
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🙏 Acknowledgements & Licenses

This project uses the following open-source components:

1.  **YOLOv12** by Ultralytics:
    -   License: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
    -   Source: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

2.  **Face Detection Weights** inspired by YapaLab:
    -   License: [GPL-3.0](https://github.com/YapaLab/yolo-face/blob/dev/LICENSE)
    -   Source: [https://github.com/YapaLab/yolo-face](https://github.com/YapaLab/yolo-face)
      
**Project License:**
This entire project is licensed under the AGPL-3.0 to comply with the licensing terms of the YOLO ecosystem.

## 💬 Support & Q&A

[![Have a question?](https://img.shields.io/badge/Ask%20a%20Question-GitHub%20Discussions-blue?style=for-the-badge&logo=github)](https://github.com/RevDra/human-face-detection/discussions)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/luongminhngoc)

---

**Last Updated:** February 1, 2026 |
**Status:** ✅ Production Ready
