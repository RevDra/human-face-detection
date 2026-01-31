# 📦 YOLOv12 Face Detection Models

This folder contains the **YOLOv12** models optimized for Human Face Detection.

## 📊 Available Models

### ✅ Included Models (Ready to use)
These models are lightweight and already included in this repository:

1. **yolov12n-face.pt (Nano)**
   - ⚡ **Fastest** inference speed.
   - 🎯 Best for: **Live Webcam**, low-power devices (CPU).
   - 📉 Accuracy: Good for close-up faces.

2. **yolov12s-face.pt (Small)**
   - ⚖️ **Balanced** speed and accuracy.
   - 🎯 Best for: **Video Processing** where decent FPS is needed.
   - 📈 Accuracy: Better detection at medium distances.

### 📥 Models to Download (Required for higher accuracy)
Due to file size limits, the heavier models must be downloaded manually or via script:

3. **yolov12m-face.pt (Medium)**
   - 🎯 Best for: High-precision video analysis, security footage.
   - 🐢 Speed: Moderate (requires decent GPU).

4. **yolov12l-face.pt (Large)**
   - 🎯 Best for: **Static Images**, difficult lighting, small faces in large crowds.
   - 🐢 Speed: Slowest but most accurate (SOTA).

---

## ⬇️ Download Instructions

The models are sourced from the [YapaLab/yolo-face](https://github.com/YapaLab/yolo-face) repository (Release 1.0.0).

### Option 1: Automated Download (Linux/Mac)
Run the following commands in your terminal (inside the project root):

```bash
# Download Medium Model
wget [https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12m-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12m-face.pt) -P models/

# Download Large Model
wget [https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt) -P models/

### Option 2: Clone from Original Repository
```bash
# Clone the YOLOv8-Face repository
git clone https://github.com/Yusepp/YOLOv8-Face.git

# Copy models to your project
cp YOLOv8-Face/models/yolov8*.pt ./models/
```

### Option 2: Direct Download (Manual)
Click the links below to download the files and place them into the models/ directory of this project.

| Model | Size | Download Link |
|-------|------|---------------|
| Medium | ~40 MB | [yolo12m-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12m-face.pt) |
| Large | ~53 MB | [yolo12l-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt) |

## 📂 Model File Organization

After downloading, your `models/` folder should look like:
```
models/
├── yolov12n-face.pt      # ✅ Already included (6 MB)
├── yolov12s-face.pt      # ✅ Already included (18 MB)
├── yolov12m-face.pt      # 📥 Download needed (40 MB)
├── yolov12l-face.pt      # 📥 Download needed (52 MB)
└── MODELS.md             # This file
```

## How to Use

Once all models are downloaded, you can select them from the web interface:

1. Open `http://localhost:7860` in your browser
2. Go to the **Image**, **Video**, or **Webcam** tab
3. Select desired model from the dropdown:
   - **Nano (n)** - Fastest (default)
   - **Small (s)** - Balanced
   - **Medium (m)** - High Precious 
   - **Large (l)** - Max Accuracy

## 📈 Model Comparison

| Model | Suffix | Best Use Case | Inference Speed | Accuracy |
|-------|--------|---------------|-----------------|----------|
| Nano | n | Webcam | CPU | 🚀 Very Fast | ⭐⭐⭐
| Small | s | Standard Video | 🚀 Fast | ⭐⭐⭐⭐
| Medium | m | Analysis | GPU | 🐢 Moderate | ⭐⭐⭐⭐⭐
| Large | l | High-Res Images | 🐢 Slow | ⭐⭐⭐⭐⭐⭐

## 🐛 Troubleshooting

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

### "RuntimeError: PytorchStreamReader..."
**Solution:**
- Ensure to be connected to the Internet
- Delete the .pt file and download it again

## References

- [YOLOv12 GitHub Repository](https://github.com/YapaLab/yolo-face)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyPI Ultralytics](https://pypi.org/project/ultralytics/)

---

**Note:** The medium and large models in this project are from the YOLOv8-Face repository, which are specifically optimized for face detection. For generic object detection models, refer to the official Ultralytics documentation.


