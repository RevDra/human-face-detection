from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List

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

        dummy_input = np.zeros(
            (1, 3, self.input_height, self.input_width), dtype=np.float32
        )
        self.session.run(None, {self.input_name: dummy_input})

        self._initialized = True

    def _preprocess(self, image: np.ndarray):
        """Preprocess Letterbox utilizing given cache and Fused C++ blobFromImage"""
        img_h, img_w = image.shape[:2]
        scale = min(self.input_size / img_w, self.input_size / img_h)
        new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))

        pad_w = (self.input_size - new_w) / 2.0
        pad_h = (self.input_size - new_h) / 2.0
        top = int(round(pad_h - 0.1))
        left = int(round(pad_w - 0.1))

        if top > 0:
            self._letterbox_buffer[:top, :] = 114
        if top + new_h < self.input_height:
            self._letterbox_buffer[top + new_h :, :] = 114
        if left > 0:
            self._letterbox_buffer[top : top + new_h, :left] = 114
        if left + new_w < self.input_width:
            self._letterbox_buffer[top : top + new_h, left + new_w :] = 114

        cv2.resize(
            image,
            (new_w, new_h),
            dst=self._letterbox_buffer[top : top + new_h, left : left + new_w],
            interpolation=cv2.INTER_LINEAR,
        )

        blob = cv2.dnn.blobFromImage(
            self._letterbox_buffer,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            swapRB=True,
            crop=False,
        )

        return blob, scale, left, top

    def detect_faces(self, image, conf_threshold=0.32, iou_threshold=0.45):
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        blob, scale, pad_left, pad_top = self._preprocess(image)

        raw_output = self.session.run(None, {self.input_name: blob})[0]

        predictions = np.squeeze(raw_output)
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        scores = (
            np.max(predictions[:, 4:], axis=1)
            if predictions.shape[1] > 5
            else predictions[:, 4]
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
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(), conf_threshold, iou_threshold
        )

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

    def draw_faces(
        self, image, detections, show_conf=True, blur_faces=False, inplace=False
    ):
        result = image if inplace else image.copy()

        for det in detections:
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]
            conf = det["confidence"]

            if blur_faces:
                face_roi = result[y1:y2, x1:x2]
                if face_roi.size > 0:
                    blur_w = max(15, (x2 - x1) // 3 | 1)
                    blur_h = max(15, (y2 - y1) // 3 | 1)
                    result[y1:y2, x1:x2] = cv2.GaussianBlur(
                        face_roi, (blur_w, blur_h), 30
                    )

            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            if show_conf:
                label = f"Face {conf:.2f}"
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
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
        cv2.putText(
            result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
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


def _compute_iou(boxA: Dict[str, Any], boxB: Dict[str, Any]) -> float:
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    areaA = max(1, boxA["w"] * boxA["h"])
    areaB = max(1, boxB["w"] * boxB["h"])
    return inter_area / float(areaA + areaB - inter_area)


def _interpolate_detections(
    dets_start: List[Dict[str, Any]],
    dets_end: List[Dict[str, Any]],
    alpha: float,
    iou_match_thresh: float = 0.25,
) -> List[Dict[str, Any]]:
    if not dets_start:
        return dets_end if alpha > 0.5 else []
    if not dets_end:
        return dets_start if alpha <= 0.5 else []

    interpolated = []
    used_end = set()

    for d_start in dets_start:
        best_iou = 0.0
        best_idx = -1
        for idx, d_end in enumerate(dets_end):
            if idx in used_end:
                continue
            iou = _compute_iou(d_start, d_end)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx != -1 and best_iou >= iou_match_thresh:
            used_end.add(best_idx)
            d_end = dets_end[best_idx]
            x1 = int(round((1.0 - alpha) * d_start["x1"] + alpha * d_end["x1"]))
            y1 = int(round((1.0 - alpha) * d_start["y1"] + alpha * d_end["y1"]))
            x2 = int(round((1.0 - alpha) * d_start["x2"] + alpha * d_end["x2"]))
            y2 = int(round((1.0 - alpha) * d_start["y2"] + alpha * d_end["y2"]))
            conf = (1.0 - alpha) * d_start["confidence"] + alpha * d_end["confidence"]
            interpolated.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "confidence": conf,
                }
            )
        elif alpha <= 0.5:
            interpolated.append(d_start)

    return interpolated


def detect_from_image(image_path, model_path, output_path=None, conf_threshold=0.32):
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    detector = YOLOv12FaceDetector(model_path)
    detections = detector.detect_faces(image, conf_threshold)
    result = detector.draw_faces(image, detections, inplace=True)

    if output_path:
        cv2.imwrite(str(output_path), result)

    return detections


def detect_from_video(
    video_path,
    model_path,
    output_path=None,
    conf_threshold=0.32,
    frame_stride=4,
    max_dimension=1280,
):
    """
    Video Pipeline tối ưu hóa siêu tốc:
    - Adaptive Frame Stride vs Linear Box Interpolation.
    - Zero-temp-file streaming directly through FFmpeg Stdin Pipe.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    scale_factor = 1.0
    if max(orig_w, orig_h) > max_dimension:
        scale_factor = max_dimension / float(max(orig_w, orig_h))
    out_w = int(round(orig_w * scale_factor)) // 2 * 2
    out_h = int(round(orig_h * scale_factor)) // 2 * 2
    need_resize = (out_w != orig_w) or (out_h != orig_h)

    detector = YOLOv12FaceDetector(model_path)

    ffmpeg_proc = None
    cv_writer = None

    if output_path:
        if shutil.which("ffmpeg"):
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{out_w}x{out_h}",
                "-pix_fmt",
                "bgr24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            cv_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out_w, out_h),
            )

    def _read_and_prepare():
        ret, frm = cap.read()
        if not ret:
            return None
        if need_resize:
            frm = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return frm

    def _write_frame(frm: np.ndarray):
        if ffmpeg_proc and ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.write(frm.tobytes())
        elif cv_writer:
            cv_writer.write(frm)

    prev_frame = _read_and_prepare()
    if prev_frame is None:
        cap.release()
        return

    prev_dets = detector.detect_faces(prev_frame, conf_threshold)
    detector.draw_faces(prev_frame, prev_dets, inplace=True)
    _write_frame(prev_frame)

    while True:
        chunk_frames = []
        for _ in range(frame_stride):
            frm = _read_and_prepare()
            if frm is None:
                break
            chunk_frames.append(frm)

        if not chunk_frames:
            break

        last_frame = chunk_frames[-1]
        next_dets = detector.detect_faces(last_frame, conf_threshold)

        chunk_len = len(chunk_frames)
        for i in range(chunk_len - 1):
            alpha = float(i + 1) / float(chunk_len)
            interp_dets = _interpolate_detections(prev_dets, next_dets, alpha)
            detector.draw_faces(chunk_frames[i], interp_dets, inplace=True)
            _write_frame(chunk_frames[i])

        detector.draw_faces(last_frame, next_dets, inplace=True)
        _write_frame(last_frame)

        prev_dets = next_dets

    cap.release()
    if ffmpeg_proc and ffmpeg_proc.stdin:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    if cv_writer:
        cv_writer.release()
