# 🔍 YOLOv8 Human Face Detection Web Application

A professional, real-time face detection system built with YOLOv8 and Flask. Detect faces in images, videos, and live webcam streams with high accuracy.
View the demo using this [link](https://revdra-yolov8-hfd.hf.space/), which I have modified for better alignment to Hugging Face Spaces.

## ✨ Features

### 📷 Image Detection
- Upload and detect faces in images (JPG, PNG, GIF)
- Real-time detection with confidence scores
- Download annotated result images
- Display face statistics (count, position, size)

### 🎬 Video Detection
- Process video files (MP4, AVI, MOV, MKV)
- Annotate each frame with bounding boxes
- Download processed video
- Watch results directly in the web player

### 📹 Live Webcam
- Real-time detection from your webcam
- Side-by-side video feed and detection results
- Live statistics (FPS, face count, duration)
- Start/stop controls

### 🤖 Model Selection
- **YOLOv8 Nano** (yolov8n_100e.pt) - Fastest, 30-60 FPS
- **YOLOv8 Medium** (yolov8m_200e.pt) - Balanced speed/accuracy
- **YOLOv8 Large** (yolov8l_100e.pt) - Best accuracy, 5-15 FPS

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see below)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the web server:**
```bash
python src/web_app.py
```

3. **Open in browser:**
```
http://localhost:5000
```

## 📁 Project Structure

```
Human_face_detection/
├── README.md                           # Main documentation
├── requirements.txt                    # Python dependencies
│
├── src/                                # Source code
│   ├── web_app.py                     # Flask web server
│   └── face_detection_yolov8.py       # YOLOv8 detection engine
│
├── models/                             # YOLOv8 models
│   ├── yolov8n_100e.pt                # YOLOv8 Nano (fastest)
|   └── MODELS.md                      # Instruction to download YOLOv8 Medium (yolov8m_200e.pt) and Large (yolov8l_100e.pt)
│
├── web/                                # Web interface
│   └── templates/
│       └── index.html                 # Web UI (HTML/CSS/JS)
│
├── config/                             # Configuration files
    ├── Dockerfile                     # Docker configuration
    ├── docker-compose.yml             # Docker compose setup
    ├── deploy.sh                      # Linux deployment
    ├── deploy.bat                     # Windows deployment
    └── DEPLOYMENT_GUIDE.txt           # Deployment guide

```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/api/detect-image` | Detect faces in uploaded image |
| POST | `/api/detect-video` | Detect faces in uploaded video |
| GET | `/api/models` | List available models |
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
    data = {'model': 'yolov8m_200e.pt'}
    response = requests.post('http://localhost:5000/api/detect-image', 
                            files=files, data=data)
    result = response.json()
    
print(f"Detected {result['detections']['count']} faces")
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose (from project root)
docker-compose -f config/docker-compose.yml up --build

# Or build manually
docker build -t yolov8-face-detection -f config/Dockerfile .
docker run -p 5000:5000 -v $(pwd)/data:/app/data yolov8-face-detection
```

## 📊 Detection Details

### Confidence Threshold
Default: 0.35 (35%)
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
- `conf_threshold` - Detection confidence threshold (default: 0.35)

## 📝 Notes

### Model Files Required
Three model files are required in the `models/` directory:
- `yolov8n_100e.pt` (23 MB) ✅ Included
- `yolov8m_200e.pt` (197 MB) 📥 [Download](models/MODELS.md)
- `yolov8l_100e.pt` (83 MB) 📥 [Download](models/MODELS.md)

**See [models/MODELS.md](models/MODELS.md) for detailed download instructions.**

### Performance Tips
1. Use **YOLOv8 Nano** for webcam (fastest)
2. Use **YOLOv8 Medium** for videos (balanced)
3. Use **YOLOv8 Large** for images (most accurate)
4. Reduce image/video resolution for faster processing

### Browser Compatibility
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- IE: ❌ Not supported

## 🐛 Troubleshooting

### Models Not Found
```
FileNotFoundError: Model not found: models/yolov8m_200e.pt
```
**Solution:** Download missing models from [models/MODELS.md](models/MODELS.md). Only `yolov8n_100e.pt` is included by default.

### Camera Permission Denied
**Solution:** Grant camera permission in browser settings or use HTTPS

### Out of Memory
**Solution:** Use smaller model (Nano) or reduce video resolution

### Slow Detection
**Solution:** 
- Use YOLOv8 Nano
- Reduce input resolution
- Check CPU/GPU usage

## 📚 References

- [Model Setup Guide](models/MODELS.md) - Download and setup instructions
- [YOLOv8-Face Repository](https://github.com/Yusepp/YOLOv8-Face) - Original face-optimized models
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) - YOLOv8 documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🙏 Credits

Special thanks to **[Yusepp](https://github.com/Yusepp)** for the [YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) repository and the pre-trained face-optimized models used in this project.

---

**Last Updated:** January 28, 2026 |
**Status:** ✅ Production Ready
