from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from typing import Any, Dict

class YOLOv12FaceDetector:
    """
    Bộ nhận diện khuôn mặt YOLOv12 chạy thuần ONNX Runtime (Zero-PyTorch).
    Áp dụng Singleton Pattern để cache phiên làm việc (InferenceSession) trong RAM.
    """

    _models_dir = (Path(__file__).resolve().parent.parent / "models").resolve()
    _instances: dict[str, Any] = {}

        candidate_path = Path(model_path).resolve()
        try:
            candidate_path.relative_to(cls._models_dir)
        except ValueError:
            raise ValueError("Đường dẫn mô hình không hợp lệ: phải nằm trong thư mục 'models/'.")

        resolved_path = str(candidate_path)
        resolved_path = str(Path(model_path).resolve())
        if resolved_path not in cls._instances:
            instance = super(YOLOv12FaceDetector, cls).__new__(cls)
            instance._initialized = False
            cls._instances[resolved_path] = instance
        return cls._instances[resolved_path]

    def __init__(self, model_path="yolov12n-face.onnx"):
        if getattr(self, "_initialized", False):
            return
        model_file = Path(model_path).resolve()
        try:
            model_file.relative_to(self._models_dir)
        except ValueError:
            raise ValueError("Đường dẫn mô hình không hợp lệ: phải nằm trong thư mục 'models/'.")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy tệp mô hình: {model_file}\n"
                f"Hãy đặt tệp mô hình vào thư mục 'models/'."
            )

        self.model_path = model_file

        # Tối ưu hóa đồ thị tính toán ONNX ở mức tối đa
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # Tùy chỉnh theo số nhân CPU vật lý
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Tự động ưu tiên OpenVINO hoặc CUDA nếu môi trường có hỗ trợ, fallback về CPU
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

        # Lấy thông tin đầu vào của mô hình (mặc định chuẩn YOLO là 1x3x640x640)
        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        input_shape = model_inputs.shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

        self._initialized = True

    def _preprocess(self, image: np.ndarray):
        """
        Tiền xử lý ảnh chuẩn Letterbox của YOLO bằng thuần OpenCV C++:
        - Giữ nguyên tỷ lệ khung hình (aspect ratio), bù viền xám (114, 114, 114).
        - Chuyển BGR -> RGB, HWC -> NCHW và chuẩn hóa về [0.0, 1.0] qua blobFromImage.
        """
        img_h, img_w = image.shape[:2]
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))

        pad_w = (self.input_width - new_w) / 2.0
        pad_h = (self.input_height - new_h) / 2.0

        if (img_w, img_h) != (new_w, new_h):
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Tạo blob (1, 3, H, W) kiểu float32 cực nhanh bằng hàm C++ của OpenCV
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (self.input_width, self.input_height), swapRB=True, crop=False
        )
        return blob, scale, left, top

    def detect_faces(self, image, conf_threshold=0.32, iou_threshold=0.45):
        """
        Suy luận trực tiếp trên ONNX Runtime và lọc kết quả bằng Vectorization + NMS.
        """
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        blob, scale, pad_left, pad_top = self._preprocess(image)

        # Chạy suy luận trực tiếp trên C++ Engine của ONNX Runtime
        raw_output = self.session.run(None, {self.input_name: blob})[0]

        # Output của YOLOv12 cho 1 class có dạng (1, 5, 8400) hoặc (1, 8400, 5)
        predictions = np.squeeze(raw_output)
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # Đưa về dạng chuẩn (8400, 5): [cx, cy, w, h, conf]

        # Nếu mô hình có nhiều hơn 1 class, lấy max score từ cột thứ 4 trở đi
        if predictions.shape[1] > 5:
            scores = np.max(predictions[:, 4:], axis=1)
        else:
            scores = predictions[:, 4]

        # Bước 1: Lọc nhanh bằng mảng Boolean của NumPy (loại bỏ >99% các box rác)
        valid_mask = scores >= conf_threshold
        predictions = predictions[valid_mask]
        scores = scores[valid_mask]

        if len(predictions) == 0:
            return []

        # Bước 2: Khôi phục tọa độ từ không gian 640x640 về kích thước ảnh gốc
        cx = (predictions[:, 0] - pad_left) / scale
        cy = (predictions[:, 1] - pad_top) / scale
        w = predictions[:, 2] / scale
        h = predictions[:, 3] / scale

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

        # Đóng gói danh sách [x, y, w, h] cho thuật toán NMS của OpenCV
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

    def detect_faces_optimized(self, image, conf_threshold=0.32, iou_threshold=0.45, max_width=640):
        """
        Giữ lại phương thức này để tương thích hoàn toàn với app.py.
        Vì _preprocess() đã tự động resize Letterbox về 640x640 bằng C++,
        ta có thể gọi thẳng detect_faces() mà không cần resize 2 lần.
        """
        return self.detect_faces(image, conf_threshold, iou_threshold)

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

        info = f"Faces: {len(detections)} (YOLOv12 ONNX)"
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
    Xử lý video kết hợp kỹ thuật Frame Striding (mặc định nhảy bước 2 frame)
    giúp tăng gấp đôi tốc độ render video mà mắt thường không nhận ra độ trễ.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = YOLOv12FaceDetector(model_path)
    writer = (
        cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if output_path
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
