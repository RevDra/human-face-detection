import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv12FaceDetector:
    """
    YOLOv12 Face Detector on pure ONNX Runtime.
    Apply Singleton Pattern to cache Inference Session in RAM.
    """

    _instances: dict[str, Any] = {}
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _MODELS_DIR = (_PROJECT_ROOT / "models").resolve()

    @classmethod
    def _resolve_model_path(cls, model_path: str) -> Path:
        raw_value = str(model_path)
        safe_name = Path(raw_value).name
        if safe_name != raw_value:
            raise ValueError("Model path must be a file name within the models directory.")

        candidate = cls._MODELS_DIR / safe_name
        resolved = candidate.resolve()
        try:
            resolved.relative_to(cls._MODELS_DIR)
        except ValueError as exc:
            raise ValueError("Model path is outside the allowed models directory.") from exc
        return resolved

    def __new__(cls, model_path="yolov12n-face-int8.onnx"):
        resolved_path = str(cls._resolve_model_path(str(model_path)))
        if resolved_path not in cls._instances:
            instance = super(YOLOv12FaceDetector, cls).__new__(cls)
            instance._initialized = False
            cls._instances[resolved_path] = instance
        return cls._instances[resolved_path]

    def __init__(self, model_path="yolov12n-face-int8.onnx"):
        if getattr(self, "_initialized", False):
            return

        model_file = self._resolve_model_path(str(model_path))
        if not model_file.exists():
            raise FileNotFoundError(
                f"The model is not found at: {model_file}\n"
                f"Please put the model's file at folder 'models/'."
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

        available_providers = ort.get_available_providers()
        preferred_providers = [
            p
            for p in [
                "OpenVINOExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            if p in available_providers
        ]

        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=sess_options,
            providers=preferred_providers or ["CPUExecutionProvider"],
        )

        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        input_shape = model_inputs.shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

        self.input_size = self.input_height
        self._letterbox_buffer = np.full(
            (self.input_height, self.input_width, 3), 114, dtype=np.uint8
        )
        self._blob_buffer = np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32)

        dummy_input = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy_input})

        self._initialized = True

    def _preprocess(self, image):
        """Preprocess Letterbox utilizing given cache (Zero-Copy / In-place)"""
        img_h, img_w = image.shape[:2]
        scale = min(self.input_size / img_w, self.input_size / img_h)
        new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))

        pad_w = (self.input_size - new_w) / 2.0
        pad_h = (self.input_size - new_h) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        self._letterbox_buffer.fill(114)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._letterbox_buffer[top : top + new_h, left : left + new_w] = resized

        rgb = self._letterbox_buffer[:, :, ::-1].transpose(2, 0, 1)
        np.multiply(rgb, 1.0 / 255.0, out=self._blob_buffer[0], casting="unsafe")

        return self._blob_buffer, scale, left, top

    def detect_faces(self, image, conf_threshold=0.32, iou_threshold=0.45):
        """
        Direct inference on ONNX Runtime and filter results by Vectorization + NMS.
        """
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        blob, scale, pad_left, pad_top = self._preprocess(image)

        raw_output = self.session.run(None, {self.input_name: blob})[0]

        predictions = np.squeeze(raw_output)
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        scores = (
            np.max(predictions[:, 4:], axis=1) if predictions.shape[1] > 5 else predictions[:, 4]
        )

        valid_mask = scores >= conf_threshold
        predictions = predictions[valid_mask]
        scores = scores[valid_mask]

        if len(predictions) == 0:
            return []

        cx = (predictions[:, 0] - pad_left) / scale
        cy = (predictions[:, 1] - pad_top) / scale
        w = predictions[:, 2] / scale
        h = predictions[:, 3] / scale

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

        nms_boxes = np.column_stack((x1, y1, w, h)).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, iou_threshold)

        detections = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx1 = int(max(0, min(img_w, round(x1[idx]))))
                by1 = int(max(0, min(img_h, round(y1[idx]))))
                bw = int(round(w[idx]))
                bh = int(round(h[idx]))
                bx2 = int(max(0, min(img_w, bx1 + bw)))
                by2 = int(max(0, min(img_h, by1 + bh)))

                detections.append(
                    {
                        "x1": bx1,
                        "y1": by1,
                        "x2": bx2,
                        "y2": by2,
                        "confidence": float(scores[idx]),
                        "w": bx2 - bx1,
                        "h": by2 - by1,
                        "cx": (bx1 + bx2) // 2,
                        "cy": (by1 + by2) // 2,
                    }
                )

        return detections

    def draw_faces(self, image, detections, show_conf=True, blur_faces=False):
        result = image.copy()

        for det in detections:
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]
            conf = det["confidence"]

            if blur_faces:
                face_roi = result[y1:y2, x1:x2]
                if face_roi.size > 0:
                    blur_w = max(15, (x2 - x1) // 3 | 1)
                    blur_h = max(15, (y2 - y1) // 3 | 1)
                    result[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (blur_w, blur_h), 30)

            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            if show_conf:
                label = f"Face {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    result,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        info = f"Faces: {len(detections)} (YOLOv12 ONNX INT8)"
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return result

    def get_face_crops(self, image, detections, padding=10):
        crops = []
        h, w = image.shape[:2]

        for det in detections:
            x1 = max(0, det["x1"] - padding)
            y1 = max(0, det["y1"] - padding)
            x2 = min(w, det["x2"] + padding)
            y2 = min(h, det["y2"] + padding)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        return crops


def detect_from_image(image_path, model_path, output_path=None, conf_threshold=0.32):
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    detector = YOLOv12FaceDetector(model_path)
    detections = detector.detect_faces(image, conf_threshold)
    result = detector.draw_faces(image, detections)

    if output_path:
        cv2.imwrite(str(output_path), result)

    return detections


def detect_from_video(
    video_path, model_path, output_path=None, conf_threshold=0.32, frame_stride=2
):
    """
    Process videos with frame striding (2-frame steps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = YOLOv12FaceDetector(model_path)
    temp_output = str(output_path) + ".temp.mp4" if output_path else None
    writer = (
        cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if temp_output
        else None
    )

    frame_idx = 0
    cached_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            cached_detections = detector.detect_faces(frame, conf_threshold)

        result = detector.draw_faces(frame, cached_detections)
        if writer:
            writer.write(result)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    if output_path and temp_output and Path(temp_output).exists():
        if shutil.which("ffmpeg"):
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_output,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(output_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                Path(temp_output).unlink(missing_ok=True)
            except Exception:
                Path(temp_output).replace(output_path)
        else:
            Path(temp_output).replace(output_path)
