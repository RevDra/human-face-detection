"""
Human Face Detection using YOLOv12 (ONNX Runtime)
Supports image, video, and real-time webcam detection
"""

import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

DetectionDict = Dict[str, Any]
ProgressCallback = Callable[[int, int], None]

class YOLOv12FaceDetector:
    """Face detector using ONNX Runtime with YOLOv12 model."""

    def __init__(
        self,
        model_path: str = "models/yolov12n-face.onnx",
        conf_threshold: float = 0.32,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize the YOLOv12 Face Detector.

        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input resolution (width, height)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        model_file = Path(model_path)
        if not model_file.exists():
            project_root = Path(__file__).resolve().parent.parent
            model_file = project_root / model_path

        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found at: {model_file}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = min(os.cpu_count() or 4, 4)
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        opt_model_file = model_file.with_suffix(".opt.onnx")
        use_cached_opt = False
        try:
            use_cached_opt = (
                opt_model_file.exists()
                and opt_model_file.stat().st_mtime >= model_file.stat().st_mtime
            )
        except OSError:
            use_cached_opt = False

        if use_cached_opt:
            load_target = opt_model_file
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        else:
            load_target = model_file
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            if os.access(model_file.parent, os.W_OK):
                sess_options.optimized_model_filepath = str(opt_model_file)

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        try:
            self.session = ort.InferenceSession(
                str(load_target), sess_options=sess_options, providers=providers
            )
        except Exception:
            sess_options.optimized_model_filepath = ""
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.session = ort.InferenceSession(
                str(model_file), sess_options=sess_options, providers=providers
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Thread-local buffers to avoid race conditions across concurrent requests
        self._local = threading.local()

    def _get_buffers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get thread-local pre-allocated buffers for zero-copy preprocessing.
    
        Returns:
            Tuple of (letterbox_buf, blob_buffer).
        """
        target_w, target_h = self.input_size
        if not hasattr(self._local, "letterbox_buf"):
            self._local.letterbox_buf = np.full(
                (target_h, target_w, 3), 114, dtype=np.uint8
            )
            self._local.blob_buffer = np.full(
                (1, 3, target_h, target_w), 114.0 / 255.0, dtype=np.float32
            )
            self._local.last_box_dims = (0, 0)
        return self._local.letterbox_buf, self._local.blob_buffer
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLOv12 inference using ROI-only letterbox normalization.
    
        Args:
            image: Input BGR image (H, W, C).
    
        Returns:
            Tuple of (input_tensor, scale, (pad_w, pad_h)).
        """
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.input_size
    
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
    
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
    
        letterbox_buf, blob_buffer = self._get_buffers()

        if getattr(self._local, "last_box_dims", None) != (new_w, new_h):
            blob_buffer.fill(114.0 / 255.0)
            self._local.last_box_dims = (new_w, new_h)
    
        resized_view = letterbox_buf[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
        cv2.resize(
            image,
            (new_w, new_h),
            dst=resized_view,
            interpolation=cv2.INTER_LINEAR,
        )

        y_slice = slice(pad_h, pad_h + new_h)
        x_slice = slice(pad_w, pad_w + new_w)
        np.multiply(resized_view[:, :, 2], 1.0 / 255.0, out=blob_buffer[0, 0, y_slice, x_slice])
        np.multiply(resized_view[:, :, 1], 1.0 / 255.0, out=blob_buffer[0, 1, y_slice, x_slice])
        np.multiply(resized_view[:, :, 0], 1.0 / 255.0, out=blob_buffer[0, 2, y_slice, x_slice])
    
        return blob_buffer, scale, (pad_w, pad_h)
        
    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
        conf_threshold: Optional[float] = None,
    ) -> List[DetectionDict]:
        """
        Postprocess YOLOv12 outputs to extract face bounding boxes.

        Args:
            outputs: Raw model outputs
            orig_shape: Original image (height, width)
            scale: Letterbox scale factor
            pad: Letterbox (pad_w, pad_h)
            conf_threshold: Optional confidence override for thread-safe calls

        Returns:
            List of detection dictionaries with keys: x1, y1, x2, y2, w, h, confidence
        """
        thresh = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        predictions = np.squeeze(outputs[0])
    
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            return []
    
        scores = predictions[:, 4]
        mask = scores >= thresh
        if not np.any(mask):
            return []
    
        filtered_preds = predictions[mask]
        filtered_scores = scores[mask]

        w = filtered_preds[:, 2]
        h = filtered_preds[:, 3]
        x = filtered_preds[:, 0] - w * 0.5
        y = filtered_preds[:, 1] - h * 0.5
    
        nms_boxes = np.column_stack((x, y, w, h))
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, filtered_scores, thresh, self.iou_threshold
        )
        if len(indices) == 0:
            return []
    
        flat_idx = indices.flatten() if isinstance(indices, np.ndarray) else np.asarray(indices)

        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        inv_scale = 1.0 / scale
    
        sel_x = x[flat_idx]
        sel_y = y[flat_idx]
        sel_w = w[flat_idx]
        sel_h = h[flat_idx]
        sel_scores = filtered_scores[flat_idx]
    
        x1 = np.clip((sel_x - pad_w) * inv_scale, 0, orig_w)
        y1 = np.clip((sel_y - pad_h) * inv_scale, 0, orig_h)
        x2 = np.clip((sel_x + sel_w - pad_w) * inv_scale, 0, orig_w)
        y2 = np.clip((sel_y + sel_h - pad_h) * inv_scale, 0, orig_h)
    
        ix1 = np.rint(x1).astype(int)
        iy1 = np.rint(y1).astype(int)
        ix2 = np.rint(x2).astype(int)
        iy2 = np.rint(y2).astype(int)
    
        return [
            {
                "x1": int(ix1[i]),
                "y1": int(iy1[i]),
                "x2": int(ix2[i]),
                "y2": int(iy2[i]),
                "w": max(1, int(ix2[i] - ix1[i])),
                "h": max(1, int(iy2[i] - iy1[i])),
                "confidence": float(sel_scores[i]),
            }
            for i in range(len(flat_idx))
        ]

    def detect_faces(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect faces in a single BGR image.

        Args:
            image: Input BGR image
            conf_threshold: Optional confidence threshold override

        Returns:
            List of face detection dictionaries
        """
        if image is None or image.size == 0:
            return []

        orig_shape = image.shape[:2]
        input_tensor, scale, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self.postprocess(
            outputs, orig_shape, scale, pad, conf_threshold=conf_threshold
        )

    def draw_faces(
        self,
        image: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        blur_faces: bool = False,
        inplace: bool = False,
    ) -> np.ndarray:
        """
        Draw bounding boxes or apply privacy blur on detected faces.

        Args:
            image: Input BGR image
            detections: List of detections from detect_faces()
            color: Box color in BGR
            thickness: Line thickness
            show_confidence: Whether to draw confidence score label
            blur_faces: Whether to anonymize/blur faces instead of drawing boxes
            inplace: Modify the input image directly without copying (faster for video)

        Returns:
            Annotated BGR image
        """
        output = image if inplace else image.copy()
        if not detections:
            return output

        img_h, img_w = output.shape[:2]

        for det in detections:
            x1 = max(0, min(img_w - 1, det["x1"]))
            y1 = max(0, min(img_h - 1, det["y1"]))
            x2 = max(0, min(img_w, det["x2"]))
            y2 = max(0, min(img_h, det["y2"]))

            if x2 <= x1 or y2 <= y1:
                continue

            if blur_faces:
                face_roi = output[y1:y2, x1:x2]
                roi_h, roi_w = face_roi.shape[:2]
                if roi_w >= 16 and roi_h >= 16:
                    small_w = max(2, roi_w // 8)
                    small_h = max(2, roi_h // 8)
                    small = cv2.resize(
                        face_roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST
                    )
                    small = cv2.GaussianBlur(small, (5, 5), 0)
                    cv2.resize(
                        small,
                        (roi_w, roi_h),
                        dst=face_roi,
                        interpolation=cv2.INTER_LINEAR,
                    )
                elif roi_w > 0 and roi_h > 0:
                    output[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (15, 15), 0)
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

                if show_confidence:
                    conf_text = f"{det['confidence']:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    font_thick = 2
                    (text_w, text_h), baseline = cv2.getTextSize(
                        conf_text, font, font_scale, font_thick
                    )

                    label_y1 = max(0, y1 - text_h - baseline - 6)
                    label_y2 = label_y1 + text_h + baseline + 6
                    label_x2 = min(img_w, x1 + text_w + 8)

                    cv2.rectangle(
                        output, (x1, label_y1), (label_x2, label_y2), color, -1
                    )
                    cv2.putText(
                        output,
                        conf_text,
                        (x1 + 4, label_y2 - baseline - 3),
                        font,
                        font_scale,
                        (0, 0, 0),
                        font_thick,
                        cv2.LINE_AA,
                    )

        return output

    def get_face_crops(
        self, image: np.ndarray, detections: List[Dict], padding: int = 20
    ) -> List[Dict]:
        """
        Extract cropped face images with optional padding.

        Args:
            image: Input BGR image
            detections: List of face detections
            padding: Pixel padding around each face box

        Returns:
            List of dicts containing cropped 'image', 'confidence', and 'bbox'
        """
        crops = []
        img_h, img_w = image.shape[:2]

        for det in detections:
            x1 = max(0, det["x1"] - padding)
            y1 = max(0, det["y1"] - padding)
            x2 = min(img_w, det["x2"] + padding)
            y2 = min(img_h, det["y2"] + padding)

            if x2 > x1 and y2 > y1:
                face_crop = image[y1:y2, x1:x2].copy()
                crops.append(
                    {
                        "image": face_crop,
                        "confidence": det["confidence"],
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return crops

    @staticmethod
    def _prepare_interpolation_plan(
        prev_dets: Optional[List[DetectionDict]],
        next_dets: Optional[List[DetectionDict]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[Optional[DetectionDict]]]]:
        """
        Precompute matched bounding box deltas once per frame chunk in O(N * M) time.

        Args:
            prev_dets: Detections from the previous keyframe.
            next_dets: Detections from the current/next keyframe.

        Returns:
            Tuple of (base_boxes, delta_boxes, unmatched_fallbacks) or None if no interpolation needed.
        """
        if not next_dets or not prev_dets:
            return None

        prev_arr = np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]] for d in prev_dets],
            dtype=np.float32,
        )
        next_arr = np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]] for d in next_dets],
            dtype=np.float32,
        )

        pcx = (prev_arr[:, 0] + prev_arr[:, 2]) * 0.5
        pcy = (prev_arr[:, 1] + prev_arr[:, 3]) * 0.5
        ncx = (next_arr[:, 0] + next_arr[:, 2]) * 0.5
        ncy = (next_arr[:, 1] + next_arr[:, 3]) * 0.5

        dist_sq_mat = (ncx[:, None] - pcx[None, :]) ** 2 + (ncy[:, None] - pcy[None, :]) ** 2
        max_dist_sq = (
            np.maximum(next_arr[:, 2] - next_arr[:, 0], next_arr[:, 3] - next_arr[:, 1]) ** 2
            * 2.25
        )

        num_next = len(next_dets)
        base_arr = np.zeros((num_next, 5), dtype=np.float32)
        delta_arr = np.zeros((num_next, 5), dtype=np.float32)
        unmatched: List[Optional[DetectionDict]] = [None] * num_next

        avail_mask = np.ones(len(prev_dets), dtype=bool)

        for i in range(num_next):
            if not np.any(avail_mask):
                unmatched[i] = next_dets[i]
                continue

            row_dists = np.where(avail_mask, dist_sq_mat[i], np.inf)
            best_j = int(np.argmin(row_dists))

            if row_dists[best_j] <= max_dist_sq[i]:
                avail_mask[best_j] = False
                base_arr[i] = prev_arr[best_j]
                delta_arr[i] = next_arr[i] - prev_arr[best_j]
            else:
                unmatched[i] = next_dets[i]

        return base_arr, delta_arr, unmatched

    @staticmethod
    def _evaluate_interpolation_plan(
        plan: Tuple[np.ndarray, np.ndarray, List[Optional[DetectionDict]]],
        alpha: float,
    ) -> List[DetectionDict]:
        """
        Evaluate precomputed interpolation plan at weight alpha in O(N) time.
        """
        base_arr, delta_arr, unmatched = plan
        blended_all = base_arr + alpha * delta_arr
        coords_int = np.rint(blended_all[:, :4]).astype(int)

        results: List[DetectionDict] = []
        for i, fallback in enumerate(unmatched):
            if fallback is not None:
                results.append(fallback)
            else:
                ix1, iy1, ix2, iy2 = (
                    int(coords_int[i, 0]),
                    int(coords_int[i, 1]),
                    int(coords_int[i, 2]),
                    int(coords_int[i, 3]),
                )
                results.append(
                    {
                        "x1": ix1,
                        "y1": iy1,
                        "x2": ix2,
                        "y2": iy2,
                        "w": max(1, ix2 - ix1),
                        "h": max(1, iy2 - iy1),
                        "confidence": float(blended_all[i, 4]),
                    }
                )
        return results
        
    @staticmethod
    def _interpolate_detections(
        prev_dets: List[DetectionDict],
        next_dets: List[DetectionDict],
        alpha: float,
    ) -> List[DetectionDict]:
        """
        Linearly interpolate bounding boxes between two keyframes using center matching.

        Args:
            prev_dets: Detections from the previous keyframe.
            next_dets: Detections from the current/next keyframe.
            alpha: Interpolation weight in range [0.0, 1.0].

        Returns:
            List of interpolated detection dictionaries.
        """
        if not next_dets:
            return []
        if not prev_dets or alpha >= 1.0:
            return next_dets
        if alpha <= 0.0:
            return prev_dets
    
        plan = YOLOv12FaceDetector._prepare_interpolation_plan(prev_dets, next_dets)
        if plan is None:
            return next_dets
        return YOLOv12FaceDetector._evaluate_interpolation_plan(plan, alpha)

    def detect_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        blur_faces: bool = False,
        progress_callback=None,
        frame_stride: int = 4,
        conf_threshold: Optional[float] = None
    ) -> Dict:
        """
        Process a video file with 3-stage async pipeline, ring-buffer memory reuse,
        and motion-gated adaptive inference stride.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_stride = max(1, int(frame_stride))
        ffmpeg_proc = None
        out_cv2 = None

        if output_path:
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{width}x{height}",
                    "-pix_fmt",
                    "bgr24",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
                try:
                    ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    ffmpeg_proc = None

            if ffmpeg_proc is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_cv2 = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )

        num_slots = 3
        slot_buffers = [
            [np.empty((height, width, 3), dtype=np.uint8) for _ in range(frame_stride)]
            for _ in range(num_slots)
        ]

        free_slots_q: queue.Queue = queue.Queue()
        for s_idx in range(num_slots):
            free_slots_q.put(s_idx)

        decode_q: queue.Queue = queue.Queue(maxsize=2)
        encode_q: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        writer_errors: List[Exception] = []

        def _reader_worker():
            try:
                while not stop_event.is_set():
                    slot_idx = free_slots_q.get()
                    if slot_idx is None:
                        break

                    buffers = slot_buffers[slot_idx]
                    valid_count = 0

                    for i in range(frame_stride):
                        ret, frm = cap.read(buffers[i])
                        if not ret:
                            break

                        if frm is not buffers[i]:
                            if frm.shape[:2] != (height, width):
                                cv2.resize(
                                    frm, (width, height), dst=buffers[i]
                                )
                            else:
                                np.copyto(buffers[i], frm)
                        valid_count += 1

                    if valid_count == 0:
                        free_slots_q.put(slot_idx)
                        decode_q.put(None)
                        break

                    decode_q.put((slot_idx, valid_count))
            except Exception as e:
                writer_errors.append(e)
                decode_q.put(None)

        def _writer_worker():
            try:
                while True:
                    item = encode_q.get()
                    if item is None:
                        break
                    slot_idx, valid_count = item
                    buffers = slot_buffers[slot_idx]

                    for i in range(valid_count):
                        frame_arr = buffers[i]
                        if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                            ffmpeg_proc.stdin.write(memoryview(frame_arr))
                        elif out_cv2 is not None:
                            out_cv2.write(frame_arr)

                    free_slots_q.put(slot_idx)
            except Exception as e:
                writer_errors.append(e)
                stop_event.set()

        reader_thread = threading.Thread(target=_reader_worker, daemon=True)
        writer_thread = threading.Thread(target=_writer_worker, daemon=True)
        reader_thread.start()
        writer_thread.start()

        frame_count = 0
        total_detections = 0
        max_faces = 0
        start_time = time.time()

        prev_dets: Optional[List[Dict]] = None
        prev_thumb: Optional[np.ndarray] = None
        skipped_chunks = 0
        max_skip_chunks = 3
        
        thumb_bgr_buf = np.empty((32, 32, 3), dtype=np.uint8)
        prev_thumb_buf = np.empty((32, 32), dtype=np.uint8)
        curr_thumb_buf = np.empty((32, 32), dtype=np.uint8)
        has_prev_thumb = False
        try:
            while not stop_event.is_set():
                item = decode_q.get()
                if item is None:
                    break

                slot_idx, chunk_len = item
                buffers = slot_buffers[slot_idx]

                if prev_dets is None:
                    first_frame = buffers[0]
                    prev_dets = self.detect_faces(first_frame, conf_threshold=conf_threshold)
                    cv2.resize(
                        first_frame,
                        (32, 32),
                        dst=thumb_bgr_buf,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.cvtColor(thumb_bgr_buf, cv2.COLOR_BGR2GRAY, dst=prev_thumb_buf)
                    has_prev_thumb = True

                if chunk_len == 1:
                    next_dets = prev_dets
                else:
                    last_frame = buffers[chunk_len - 1]
                    cv2.resize(
                        last_frame,
                        (32, 32),
                        dst=thumb_bgr_buf,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.cvtColor(thumb_bgr_buf, cv2.COLOR_BGR2GRAY, dst=curr_thumb_buf)
                    motion_diff = (
                        float(cv2.absdiff(curr_thumb_buf, prev_thumb_buf).mean())
                        if has_prev_thumb
                        else 999.0
                    )

                    if motion_diff < 1.2 and skipped_chunks < max_skip_chunks:
                        next_dets = prev_dets
                        skipped_chunks += 1
                    else:
                        next_dets = self.detect_faces(last_frame, conf_threshold=conf_threshold)
                        np.copyto(prev_thumb_buf, curr_thumb_buf)
                        skipped_chunks = 0

                both_empty = (not prev_dets) and (not next_dets)
                interp_plan = (
                    None
                    if (both_empty or chunk_len <= 1 or prev_dets is next_dets)
                    else self._prepare_interpolation_plan(prev_dets, next_dets)
                )

                for i in range(chunk_len):
                    frm = buffers[i]
                    dets: List[DetectionDict] = []
                    if not both_empty:
                        if i == chunk_len - 1 or prev_dets is next_dets or interp_plan is None:
                            dets = next_dets or []
                        else:
                            alpha = (i + 1) / chunk_len
                            dets = self._evaluate_interpolation_plan(interp_plan, alpha)

                        if dets:
                            self.draw_faces(
                                frm, dets, blur_faces=blur_faces, inplace=True
                            )

                    frame_count += 1
                    num_faces = len(dets)
                    total_detections += num_faces
                    if num_faces > max_faces:
                        max_faces = num_faces

                    if show_preview:
                        cv2.imshow("YOLOv12 Face Detection - Video", frm)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_event.set()
                            break

                prev_dets = next_dets

                if progress_callback:
                    progress_callback(frame_count, total_frames)

                encode_q.put((slot_idx, chunk_len))

                if stop_event.is_set():
                    break

        finally:
            stop_event.set()
            while not decode_q.empty():
                try:
                    decode_q.get_nowait()
                except queue.Empty:
                    break
            try:
                free_slots_q.put_nowait(None)
            except queue.Full:
                pass
            reader_thread.join(timeout=2.0)

            try:
                encode_q.put(None, timeout=5.0)
            except queue.Full:
                pass
            writer_thread.join(timeout=15.0)

            cap.release()
            if ffmpeg_proc is not None:
                try:
                    if ffmpeg_proc.stdin:
                        ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait(timeout=15.0)
                except Exception:
                    ffmpeg_proc.kill()
            if out_cv2 is not None:
                out_cv2.release()
            if show_preview:
                cv2.destroyAllWindows()

        writer_thread.join(timeout=10.0)
        reader_thread.join(timeout=2.0)

        if writer_errors:
            raise writer_errors[0]

        elapsed = max(time.time() - start_time, 1e-6)
        avg_fps = frame_count / elapsed
        avg_faces = total_detections / frame_count if frame_count > 0 else 0.0

        return {
            "total_frames": frame_count,
            "total_detections": total_detections,
            "avg_faces_per_frame": float(avg_faces),
            "max_faces_in_frame": max_faces,
            "fps": float(avg_fps),
        }

    def detect_from_webcam(self, camera_id: int = 0, blur_faces: bool = False):
        """Run real-time face detection on local webcam stream."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID: {camera_id}")

        print("Starting webcam detection... Press 'q' to quit.")
        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detect_faces(frame)
                self.draw_faces(
                    frame, detections, blur_faces=blur_faces, inplace=True
                )

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Faces: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("YOLOv12 Real-time Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="Human Face Detection using YOLOv12 ONNX"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "video", "webcam"],
        help="Detection mode: image, video, or webcam",
    )
    parser.add_argument(
        "--input", type=str, help="Path to input image or video file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save output image or video"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera ID for webcam mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov12n-face.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable GUI preview window",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Blur detected faces (privacy mode)",
    )

    args = parser.parse_args()

    detector = YOLOv12FaceDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    if args.mode == "image":
        if not args.input:
            parser.error("--input is required for image mode")
        image = cv2.imread(args.input)
        if image is None:
            raise ValueError(f"Failed to load image: {args.input}")

        detections = detector.detect_faces(image)
        output_img = detector.draw_faces(
            image, detections, blur_faces=args.blur
        )
        print(f"Detected {len(detections)} face(s).")

        if args.output:
            cv2.imwrite(args.output, output_img)
            print(f"Saved output image to: {args.output}")

        if not args.no_preview:
            cv2.imshow("YOLOv12 Face Detection", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.mode == "video":
        if not args.input:
            parser.error("--input is required for video mode")
        stats = detector.detect_from_video(
            video_path=args.input,
            output_path=args.output,
            show_preview=not args.no_preview,
            blur_faces=args.blur,
        )
        print("Video Processing Complete:", stats)

    elif args.mode == "webcam":
        detector.detect_from_webcam(
            camera_id=args.camera, blur_faces=args.blur
        )


if __name__ == "__main__":
    main()
