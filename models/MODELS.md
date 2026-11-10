# 📦 YOLOv12 Face Detection Models (INT8 Quantized)

This folder contains the **YOLOv12** models optimized for Human Face Detection using **ONNX Runtime**, along with the training source code and benchmark results from the WIDER FACE dataset[cite: 1].

## 📊 Available Models

### ✅ Included Models (Ready to use)
These lightweight models are statically quantized to INT8 and already included in this repository:

1. **yolov12n-face.onnx (Nano)**
   - ⚡ **Fastest** inference speed.
   - 🎯 Best for: **Live Webcam**, low-power devices, real-time CPU processing.
   - 📉 Accuracy: Good for close-up faces.

2. **yolov12s-face.onnx (Small)**
   - ⚖️ **Balanced** speed and accuracy.
   - 🎯 Best for: **Video Processing** where stable FPS is needed.
   - 📈 Accuracy: Better detection at medium distances.

3. **yolov12m-face.onnx (Medium)**
   - 🎯 Best for: High-precision video analysis, security footage, severely occluded faces.
   - 🐢 Speed: Moderate.

### 📥 Models to Download (Required for higher accuracy)
Due to file size limits, the heavier INT8 quantized models must be downloaded manually or via script:

4. **yolov12l-face.onnx (Large)**
   - 🎯 Best for: **Static Images**, difficult lighting, small faces in large crowds.
   - 🐢 Speed: Slowest but most accurate (SOTA).

---

## 📂 Directory Structure & Artifacts

The `models/` directory separates the deployment ONNX weights from the training and evaluation artifacts:

```text
models/
├── yolov12n-face.onnx   # ✅ Included weight (Nano)
├── yolov12s-face.onnx   # ✅ Included weight (Small)
├── yolov12m-face.onnx   # ✅ Included weight (Medium)
├── yolov12l-face.onnx   # 📥 Download needed (Large)
├── MODELS.md                 # This file
├── fp32-to-int8-static-quantization-of-yolov12-face.ipynb # Quantization Notebook
└── training/                 # 🧠 Training Source & Evaluation
    ├── train_yolov12-face_widerface.ipynb  # Jupyter Notebook for reproducing training
    └── benchmarks/           # 📊 Evaluation metrics and plots
        ├── combined_results.png
        ├── combined_F1_curves.png
        ├── combined_confusion_matrix.png
        ├── nano/             # Artifacts for Nano variant
        │   ├── args.yaml     # Hyperparameters configuration
        │   ├── results.csv   # Epoch-by-epoch training metrics
        │   └── ...
        ├── small/            # Artifacts for Small variant
        ├── medium/           # Artifacts for Medium variant
        └── large/            # Artifacts for Large variant

```

---

## 🏋️ Training & Quantization

The `models/` directory ensures full transparency and reproducibility:

1. **Training Notebook:** `training/train_yolov12-face_widerface.ipynb` contains the code used to train these models on the WIDER FACE dataset. *(Note: Raw WIDER FACE images are not included due to size limits).*


2. **Quantization Notebook:** `fp32-to-int8-static-quantization-of-yolov12-face.ipynb` demonstrates how FP32 PyTorch models were converted into optimized static INT8 ONNX models for fast CPU execution via ONNX Runtime.
3. **Hyperparameters (`args.yaml`):** Each model's folder inside `benchmarks/` contains the exact configuration used during training.

---

## ⬇️ Download Instructions

The models are sourced from the [YapaLab/yolo-face](https://github.com/YapaLab/yolo-face) repository.

### Option 1: Automated Download (Linux/Mac)

Run the following commands in your terminal (inside the project root):

```bash
# Download Large Model (FP32 PyTorch)
wget [https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt) -P models/

```

### Option 2: Direct Download (Manual)

Click the links below to download the files and place them directly into the `models/` directory of this project.

| Model | Size | Download Link |

|-------|------|---------------|

| Large | ~53 MB | [yolo12l-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt) |

---

## 🚀 How to Use in the Web App

Once all models are downloaded, you can select them from the FastAPI web interface:

1. Open `http://localhost:7860` in your browser


2. Go to the **Image**, **Video**, or **Webcam** tab
3. Select your desired model from the dropdown:
* **Nano (yolov12n)** - Fastest (default)
* **Small (yolov12s)** - Balanced
* **Medium (yolov12m)** - High Precision
* **Large (yolov12l)** - Max Accuracy



---

## 📈 Model Comparison

| Model | Suffix | Best Use Case | Inference Speed | Accuracy |
| --- | --- | --- | --- | --- |
| Nano | n | Webcam CPU | 🚀 Very Fast | ⭐⭐⭐ |
| Small | s | Standard Video | 🚀 Fast | ⭐⭐⭐⭐ |
| Medium | m | Analysis GPU/CPU | 🐢 Moderate | ⭐⭐⭐⭐⭐ |
| Large | l | High-Res Images | 🐢 Slow | ⭐⭐⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting

### Model Not Found Error

```text
FileNotFoundError: Model not found: models/yolov12m-face-int8.onnx

```

**Solution:** Download the model using one of the methods above.

### Download Too Slow

**Solution:**

* Use a VPN or check your network connection.
* Try downloading during off-peak hours or use Option 2 (manual download).

### Not Enough Disk Space

**Solution:**

* Ensure you have at least 500 MB of free space.
* Download one model at a time.

---

## References

* [YOLOv12 GitHub Repository](https://github.com/YapaLab/yolo-face)
* [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
* [Ultralytics YOLOv12](https://docs.ultralytics.com/)

---

**Note:** The models in this project use static INT8 quantization and run on ONNX Runtime for optimal cross-platform asynchronous performance under FastAPI.
