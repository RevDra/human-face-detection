"""
Human Face Detection using YOLOv12 with Live Camera Display
Shows real-time face detection with confidence scores
Optimized for high FPS performance using YOLOv12 Attention Mechanism
"""

import torch

try:
    from ultralytics.nn.tasks import DetectionModel

    torch.serialization.add_safe_globals([DetectionModel])
    torch.serialization.safe_globals = lambda: None
except Exception:
    pass

import time
import tkinter as tk
from pathlib import Path
from tkinter import Label

import cv2
from PIL import Image, ImageTk


class YOLOv12FaceDetector:
    """Face detection using YOLOv12 (Class name kept for compatibility)"""

    def __init__(self, model_path):
        try:
            from ultralytics import YOLO

            self.yolo = YOLO(model_path)
            print(f"Loaded YOLOv12 model from {model_path}")
        except ImportError:
            raise ImportError("ultralytics package not found. pip install ultralytics")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def detect_faces(self, image, conf_threshold=0.5):
        """Standard detection"""
        detections = []
        try:
            results = self.yolo(image, conf=conf_threshold, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": conf,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "cx": (x1 + x2) // 2,
                            "cy": (y1 + y2) // 2,
                        }
                    )
        except Exception as e:
            print(f"Detection error: {e}")
        return detections

    def detect_faces_optimized(self, image, conf_threshold=0.5, max_width=640):
        """Optimized detection with resize"""
        detections = []
        try:
            height, width = image.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_height = int(height * scale)
                inference_image = cv2.resize(image, (max_width, new_height))
            else:
                scale = 1.0
                inference_image = image

            results = self.yolo(inference_image, conf=conf_threshold, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    if scale < 1.0:
                        x1, y1, x2, y2 = (
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale),
                        )

                    detections.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": conf,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "cx": (x1 + x2) // 2,
                            "cy": (y1 + y2) // 2,
                        }
                    )
        except Exception as e:
            print(f"Detection error: {e}")
        return detections

    def draw_faces(self, image, detections, color=(0, 255, 0), thickness=2, show_confidence=True):
        result = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["confidence"]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            if show_confidence:
                conf_text = f"{conf:.1%}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    result,
                    (x1, y1 - text_size[0][1] - 10),
                    (x1 + text_size[0][0] + 5, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    result,
                    conf_text,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
        return result


# --- GUI Class (Webcam) ---
class WebcamFaceDetectionGUI:
    def __init__(self, model_path, conf_threshold=0.5, skip_frames=2, inference_width=640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.skip_frames = skip_frames
        self.inference_width = inference_width
        self.detector = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self.detection_fps = 0
        self.last_time = time.time()
        self.last_detection_time = time.time()
        self.current_detections = None

        try:
            self.detector = YOLOv12FaceDetector(model_path)
        except Exception as e:
            print(f"Error: {e}")
            return

        self.root = tk.Tk()
        self.root.title(f"YOLOv12 Face Detection - {Path(model_path).stem}")
        self.root.geometry("1024x768")

        # UI Setup
        info_frame = tk.Frame(self.root, bg="lightgray")
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.info_label = Label(
            info_frame,
            text="Initializing...",
            bg="lightgray",
            fg="black",
            font=("Arial", 10),
        )
        self.info_label.pack()

        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(self.root, bg="lightgray")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.status_label = Label(
            control_frame,
            text="Running YOLOv12",
            bg="lightgray",
            fg="green",
            font=("Arial", 10, "bold"),
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        self.quit_button = tk.Button(
            control_frame, text="Quit", command=self.on_quit, bg="red", fg="white"
        )
        self.quit_button.pack(side=tk.RIGHT, padx=5)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.config(text="Error: Could not open webcam", fg="red")
            return

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        should_detect = (self.frame_count % self.skip_frames) == 0
        if should_detect:
            detections = self.detector.detect_faces_optimized(
                frame, self.conf_threshold, self.inference_width
            )
            self.current_detections = detections
            self.detection_count += 1
            if time.time() - self.last_detection_time >= 1.0:
                self.detection_fps = self.detection_count / (time.time() - self.last_detection_time)
                self.detection_count = 0
                self.last_detection_time = time.time()
        else:
            detections = self.current_detections if self.current_detections else []

        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["confidence"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result,
                f"{conf:.1%}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        self.frame_count += 1
        if time.time() - self.last_time >= 1.0:
            self.fps = self.frame_count / (time.time() - self.last_time)
            self.frame_count = 0
            self.last_time = time.time()

        info_text = (
            f"FPS: {self.fps:.1f} | Det FPS: {self.detection_fps:.1f} | Faces: {len(detections)}"
        )
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        h, w = result.shape[:2]
        display_w = 1000
        if w > display_w:
            display_h = int(h * display_w / w)
            result = cv2.resize(result, (display_w, display_h))

        rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
        self.video_label.imgtk = tk_image
        self.video_label.config(image=tk_image)

        self.root.after(1, self.update_frame)

    def on_quit(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# --- Helper Functions ---
def webcam_detection_with_display(
    model_path, conf_threshold=0.5, skip_frames=2, inference_width=640
):
    gui = WebcamFaceDetectionGUI(model_path, conf_threshold, skip_frames, inference_width)
    gui.run()


def detect_from_video(video_path, model_path, output_path=None, conf_threshold=0.5):
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_faces(frame, conf_threshold)
        result = detector.draw_faces(frame, detections)
        if writer:
            writer.write(result)

    cap.release()
    if writer:
        writer.release()
    print("Video processing complete")


def detect_from_image(image_path, model_path, output_path=None, conf_threshold=0.5):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []

    detector = YOLOv12FaceDetector(model_path)
    detections = detector.detect_faces(image, conf_threshold)
    result = detector.draw_faces(image, detections)

    print(f"✓ Found {len(detections)} faces")
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Output saved to {output_path}")
    return result, detections


def display_image_with_detections(image, title="Result"):
    root = tk.Tk()
    root.title(title)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > 1200:
        scale = 1200 / w
        rgb = cv2.resize(rgb, (1200, int(h * scale)))
    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    label = Label(root, image=photo)
    label.image = photo
    label.pack()
    root.mainloop()


# --- Main Menu Logic ---
def select_model_level(script_dir):
    print("\n" + "-" * 40)
    print("  SELECT MODEL SIZE (YOLOv12):")
    print("  1. Nano   (yolov12n) - Siêu nhanh, nhẹ")
    print("  2. Small  (yolov12s) - Cân bằng")
    print("  3. Medium (yolov12m) - Chính xác cao")
    print("  4. Large  (yolov12l) - Chính xác nhất (Nặng)")
    print("-" * 40)

    mapping = {
        "1": "yolov12n-face.pt",
        "2": "yolov12s-face.pt",
        "3": "yolov12m-face.pt",
        "4": "yolov12l-face.pt",
    }

    while True:
        choice = input("  >> Select model (1-4) [Default=2]: ").strip()
        if choice == "":
            choice = "2"

        if choice in mapping:
            model_name = mapping[choice]
            model_path = script_dir / model_name
            if model_path.exists():
                return str(model_path)
            else:
                print(f"  [!] Error: File {model_name} not found in {script_dir}")
        else:
            print("  [!] Invalid choice.")


def choose_and_process_file(script_dir):
    while True:
        print("\n" + "█" * 70)
        print("█" + "  YOLOv12 Face Detection System - ADVANCED MODE  ".center(68) + "█")
        print("█" * 70)
        print("  1) WEBCAM        (Live Detection)")
        print("  2) VIDEO FILE    (Process Video)")
        print("  3) IMAGE FILE    (Process Image)")
        print("  0) EXIT")
        print("=" * 70)

        task_choice = input("  >> Choose task (0-3): ").strip()

        if task_choice == "0":
            break

        if task_choice in ["1", "2", "3"]:
            model_path = select_model_level(script_dir)

            if task_choice == "1":
                is_heavy = "l-face" in model_path or "m-face" in model_path
                skip = 3 if is_heavy else 1
                webcam_detection_with_display(model_path, conf_threshold=0.35, skip_frames=skip)
            elif task_choice == "2":
                video_path = input("\n  >> Path to video file: ").strip().strip("\"'")
                detect_from_video(video_path, model_path, "output_video.mp4", 0.35)
            elif task_choice == "3":
                image_path = input("\n  >> Path to image file: ").strip().strip("\"'")
                result, _ = detect_from_image(image_path, model_path, "output_image.jpg", 0.35)
                if result is not None:
                    display_image_with_detections(result)
        else:
            print("Invalid option!")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    required_models = [
        "yolov12n-face.pt",
        "yolov12s-face.pt",
        "yolov12m-face.pt",
        "yolov12l-face.pt",
    ]

    print("Checking models...")
    found_any = False
    for m in required_models:
        if (script_dir / m).exists():
            print(f"  [OK] Found {m}")
            found_any = True
        else:
            print(f"  [MISSING] {m}")

    if not found_any:
        print("\nERROR: No YOLOv12 models found! Please place .pt files in this folder.")
    else:
        try:
            choose_and_process_file(script_dir)
        except Exception as e:
            print(f"\nCritical Error: {e}")
