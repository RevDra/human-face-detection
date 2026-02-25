# 📦 YOLOv12 Face Detection Models

This folder contains the **YOLOv12** models optimized for Human Face Detection, along with the source code and benchmark results from the training process on the WIDER FACE dataset.

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

## 📂 Directory Structure & Artifacts

The `models/` directory is organized to separate the deployment weights from the training and evaluation artifacts:

```text
models/
├── yolov12n-face.pt      # ✅ Included weight (Nano)
├── yolov12s-face.pt      # ✅ Included weight (Small)
├── yolov12m-face.pt      # 📥 Download needed (Medium)
├── yolov12l-face.pt      # 📥 Download needed (Large)
├── MODELS.md             # This file
└── training/             # 🧠 Training Source & Evaluation
    ├── train_yolov12-face_widerface.ipynb  # Jupyter Notebook for reproducing the training
    └── benchmarks/                    # 📊 Evaluation metrics and plots
        ├── combined_results.png       # Combined training loss/mAP charts
        ├── combined_F1_curves.png     # Combined F1-score comparison grids
        ├── combined_confusion...      # Combined Confusion Matrices (2x2 grid)
        ├── nano/                      # Artifacts for Nano variant
        │   ├── args.yaml              # Hyperparameters configuration
        │   ├── results.csv            # Epoch-by-epoch training metrics
        │   ├── results.png            # YOLO default training overview
        │   ├── confusion_matrix.png   
        │   ├── confusion_matrix_normalized.png
        │   ├── BoxF1_curve.png        # F1-Confidence curve
        │   ├── BoxP_curve.png         # Precision curve
        │   ├── BoxR_curve.png         # Recall curve
        │   └── BoxPR_curve.png        # Precision-Recall curve
        ├── small/                     # Artifacts for Small variant
        │   └── [Same structure as nano]
        ├── medium/                    # Artifacts for Medium variant
        │   └── [Same structure as nano]
        └── large/                     # Artifacts for Large variant
            └── [Same structure as nano]
```

## 🏋️ Training & Reproducibility

The `training/` directory ensures full transparency of this project:

1. **Notebook:** `train_yolov12-face_widerface.ipynb` contains the exact code used to train these models on the WIDER FACE dataset. *(Note: The raw WIDER FACE images are not included due to size and licensing limits. Please download them from the official source if you wish to retrain)*.

2. **Hyperparameters (`args.yaml`):** Each model's folder inside `benchmarks/` contains the exact configuration used during training.

3. **Metrics & Visuals:** The `benchmarks/` folder includes comprehensive curves (F1, PR) and normalized confusion matrices to validate the model's robustness and background suppression capabilities.
   
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
```

### Option 2: Direct Download (Manual)
Click the links below to download the files and place them into the models/ directory of this project.

| Model | Size | Download Link |
|-------|------|---------------|
| Medium | ~40 MB | [yolo12m-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12m-face.pt) |
| Large | ~53 MB | [yolo12l-face.pt](https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt) |

### Option 3: Train from Scratch (Reproducibility)
If you prefer to train the models yourself to generate the `.pt` files, you can use the provided Jupyter Notebook and configuration files.

**⚠️ Note:** A CUDA-enabled GPU is highly recommended for this process.

**Steps:**
1. **Prepare the Dataset:** Download the WIDER FACE dataset (via Kaggle or the official site) and extract it into a `datasets/WIDER_FACE/` directory at the root of your project.
2. **Open the Notebook:** Navigate to the `models/training/` directory and launch `train_yolov12-face_widerface.ipynb`.
3. **Configure Hyperparameters:** You can reference the exact training configurations used for each model by checking the `args.yaml` files located inside the `models/training/benchmarks/<model_size>/` directories.
4. **Run Training:** Execute the cells in the notebook. 
5. **Move the Weights:** Once the training finishes, Ultralytics YOLO will save the best weights at `runs/detect/train/weights/best.pt`. Rename this file (e.g., to `yolov12m-face.pt`) and move it directly into the `models/` folder.

## 🚀 How to Use in the Web App

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
| Nano | n | Webcam CPU | 🚀 Very Fast | ⭐⭐⭐ |
| Small | s | Standard Video | 🚀 Fast | ⭐⭐⭐⭐ |
| Medium | m | Analysis GPU | 🐢 Moderate | ⭐⭐⭐⭐⭐ |
| Large | l | High-Res Images | 🐢 Slow | ⭐⭐⭐⭐⭐⭐ |

## 🐛 Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model not found: models/yolov12m-face.pt
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
