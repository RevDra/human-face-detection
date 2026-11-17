# 📦 YOLO26 Face Detection Models (FP32 NMS-Free ONNX)

This folder contains the **YOLO26** models optimized for Human Face Detection using **ONNX Runtime**, fine-tuned on the WIDER FACE dataset.

## 📊 Available Models

All four **YOLO26** variants are exported in **FP32** with native **NMS-free** inference (`nms=False` / one-to-one head producing `(1, 300, 6)` tensors of `[x1, y1, x2, y2, confidence, class_id]` or raw one-to-many `(1, 5, 8400)` automatically handled by `YOLO26FaceDetector`):

1. **yolo26n-face.onnx (Nano)**
   - ⚡ **Fastest** inference speed (DFL-free + NMS-free).
   - 🎯 Best for: **Live Webcam**, low-power devices, real-time CPU processing.
   - 📉 Accuracy: Great for close-up and medium-range faces.

2. **yolo26s-face.onnx (Small)**
   - ⚖️ **Balanced** speed and accuracy.
   - 🎯 Best for: **Video Processing** where stable high FPS is needed.
   - 📈 Accuracy: Strong detection at medium distances.

3. **yolo26m-face.onnx (Medium)**
   - 🎯 Best for: High-precision video analysis, security footage, occluded faces.
   - 🐢 Speed: Moderate.

4. **yolo26l-face.onnx (Large)**
   - 🎯 Best for: **Static Images**, difficult lighting, tiny faces in dense crowds.
   - 🐢 Speed: Slowest but highest accuracy (SOTA).

---

## 📂 Directory Structure & Artifacts

The `models/` directory organizes the deployment ONNX weights alongside training and evaluation artifacts:

```text
models/
├── yolo26n-face.onnx         # ✅ Nano FP32 ONNX weight (NMS-free)
├── yolo26s-face.onnx         # ✅ Small FP32 ONNX weight (NMS-free)
├── yolo26m-face.onnx         # ✅ Medium FP32 ONNX weight (NMS-free)
├── yolo26l-face.onnx         # ✅ Large FP32 ONNX weight (NMS-free)
├── MODELS.md                 # This documentation file
└── training/                 # 🧠 Training Source & Evaluation
    └── benchmarks/           # 📊 Evaluation metrics and plots
```

---

## 🚀 How to Use in the Web App

Place all 4 files (`yolo26n-face.onnx`, `yolo26s-face.onnx`, `yolo26m-face.onnx`, `yolo26l-face.onnx`) inside the `models/` directory, then start the server:

1. Open `http://localhost:7860` in your browser.
2. Go to the **Image**, **Video**, or **Webcam** tab.
3. Select your desired model from the dropdown:
   * **Nano (`yolo26n`)** - Fastest (default for Webcam)
   * **Small (`yolo26s`)** - Balanced (default for Video)
   * **Medium (`yolo26m`)** - High Precision
   * **Large (`yolo26l`)** - Max Accuracy (default for Image)

---

## 📈 Model Comparison

| Model | Key | Best Use Case | Precision | Architecture | Inference Speed | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Nano | `yolo26n` | Webcam CPU | FP32 | NMS-Free | 🚀 Very Fast | ⭐⭐⭐ |
| Small | `yolo26s` | Standard Video | FP32 | NMS-Free | 🚀 Fast | ⭐⭐⭐⭐ |
| Medium | `yolo26m` | Analysis GPU/CPU | FP32 | NMS-Free | 🐢 Moderate | ⭐⭐⭐⭐⭐ |
| Large | `yolo26l` | High-Res Images | FP32 | NMS-Free | 🐢 Slow | ⭐⭐⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting

### Model Not Found Error

```text
FileNotFoundError: ONNX model not found at: models/yolo26n-face.onnx
```

**Solution:** Ensure `yolo26n-face.onnx`, `yolo26s-face.onnx`, `yolo26m-face.onnx`, and `yolo26l-face.onnx` are placed directly inside the `models/` folder.

---

## References

* [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
* [YOLO26 End-to-End NMS-Free Detection](https://docs.ultralytics.com/guides/end2end-detection/)
* [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
