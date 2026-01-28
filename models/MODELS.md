# YOLOv8 Models

This folder contains YOLOv8 pre-trained models for face detection.

## Available Models

### 📦 Included Models
- **yolov8n_100e.pt** (23 MB) ✅ Included in repository
  - Fastest, real-time inference
  - Ideal for webcam detection (30-60 FPS)
  - Lowest accuracy

### 📥 Models to Download
The following models are **NOT included** due to GitHub file size limits (100MB max):

#### 1. YOLOv8 Medium (197 MB)
**yolov8m_200e.pt** - Balanced speed and accuracy
- Good for video processing (15-30 FPS)
- Medium accuracy
- Recommended for most use cases

#### 2. YOLOv8 Large (83 MB)
**yolov8l_100e.pt** - Highest accuracy
- Best detection accuracy
- Slower inference (5-15 FPS)
- Best for image analysis

## Download Instructions

These models are from the [YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) repository, specifically trained for face detection.

### Option 1: Direct Download from Google Drive

**YOLOv8 Medium (197 MB):**
```
https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing
```
1. Open the link in your browser
2. Click "Download" button
3. Save as `yolov8m_200e.pt` in the `models/` folder

**YOLOv8 Large (83 MB):**
```
https://drive.google.com/file/d/1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU/view?usp=sharing
```
1. Open the link in your browser
2. Click "Download" button
3. Save as `yolov8l_100e.pt` in the `models/` folder

### Option 2: Clone from Original Repository
```bash
# Clone the YOLOv8-Face repository
git clone https://github.com/Yusepp/YOLOv8-Face.git

# Copy models to your project
cp YOLOv8-Face/models/yolov8*.pt ./models/
```

### Option 3: Using gdown (Command Line)
```bash
# Install gdown
pip install gdown

# Download Medium model
gdown https://drive.google.com/uc?id=1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps- -O models/yolov8m_200e.pt

# Download Large model
gdown https://drive.google.com/uc?id=1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU -O models/yolov8l_100e.pt
```

## Model File Organization

After downloading, your `models/` folder should look like:
```
models/
├── yolov8n_100e.pt      # ✅ Already included (23 MB)
├── yolov8m_200e.pt      # 📥 Download needed (197 MB)
├── yolov8l_100e.pt      # 📥 Download needed (83 MB)
└── MODELS.md            # This file
```

## How to Use

Once all models are downloaded, you can select them from the web interface:

1. Open `http://localhost:5000` in your browser
2. Go to the **Image**, **Video**, or **Webcam** tab
3. Select desired model from the dropdown:
   - **yolov8n_100e** - Fastest (default)
   - **yolov8m_200e** - Balanced
   - **yolov8l_100e** - Most accurate

## Model Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| Nano | 23 MB | 30-60 FPS | Lower | Webcam, Real-time |
| Medium | 197 MB | 15-30 FPS | Medium | Videos |
| Large | 83 MB | 5-15 FPS | Highest | Images, Accuracy |

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model not found: models/yolov8m_200e.pt
```
**Solution:** Download the model using one of the methods above

### Download Too Slow
**Solution:** 
- Use a VPN or check your internet connection
- Try downloading during off-peak hours
- Use Option 2 (manual download) instead

### Not Enough Disk Space
**Solution:**
- Ensure you have at least 500 MB free space
- Download one model at a time
- Delete old models if needed

## References

- [YOLOv8-Face GitHub Repository](https://github.com/Yusepp/YOLOv8-Face) - Original repository with face-optimized models
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyPI Ultralytics](https://pypi.org/project/ultralytics/)

---

**Note:** The medium and large models in this project are from the YOLOv8-Face repository, which are specifically optimized for face detection. For generic object detection models, refer to the official Ultralytics documentation.
