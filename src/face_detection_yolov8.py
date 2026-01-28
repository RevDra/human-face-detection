"""
Human Face Detection using YOLOv8 with Live Camera Display
Shows real-time face detection with confidence scores
Optimized for high FPS performance
"""

import cv2
from pathlib import Path
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
import threading


class YOLOv8FaceDetector:
    """Face detection using YOLOv8"""
    
    def __init__(self, model_path):
        """
        Initialize YOLOv8 face detector
        
        Args:
            model_path: Path to YOLOv8 model file
        """
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(model_path)
            print(f"✓ Loaded YOLOv8 model from {model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install it with:\n"
                "pip install ultralytics"
            )
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def detect_faces(self, image, conf_threshold=0.5):
        """
        Detect faces in an image using YOLOv8
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections with coordinates and confidence
        """
        detections = []
        
        try:
            # Run inference
            results = self.yolo(image, conf=conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    # Get coordinates in xyxy format
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': conf,
                        'w': x2 - x1,
                        'h': y2 - y1,
                        'cx': (x1 + x2) // 2,
                        'cy': (y1 + y2) // 2
                    })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def detect_faces_optimized(self, image, conf_threshold=0.5, max_width=640):
        """
        Detect faces with optimized inference (resized input)
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            max_width: Maximum width for inference (smaller = faster)
            
        Returns:
            List of detections with coordinates and confidence
        """
        detections = []
        
        try:
            height, width = image.shape[:2]
            
            # Resize for faster inference if needed
            if width > max_width:
                scale = max_width / width
                new_height = int(height * scale)
                inference_image = cv2.resize(image, (max_width, new_height))
            else:
                scale = 1.0
                inference_image = image
            
            # Run inference
            results = self.yolo(inference_image, conf=conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    # Get coordinates in xyxy format
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Scale back to original image size if resized
                    if scale < 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    
                    detections.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': conf,
                        'w': x2 - x1,
                        'h': y2 - y1,
                        'cx': (x1 + x2) // 2,
                        'cy': (y1 + y2) // 2
                    })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def draw_faces(self, image, detections, color=(0, 255, 0), thickness=2, show_confidence=True):
        """
        Draw rectangles around detected faces with confidence scores
        
        Args:
            image: Input image
            detections: List of detections
            color: Rectangle color (BGR format)
            thickness: Rectangle line thickness
            show_confidence: Whether to show confidence score
            
        Returns:
            Image with drawn rectangles
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            if show_confidence:
                conf_text = f"{conf:.1%}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_w, text_h = text_size[0]
                
                # Draw background for text
                cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
                
                # Draw text
                cv2.putText(result, conf_text, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result


class WebcamFaceDetectionGUI:
    """GUI for real-time webcam face detection with tkinter"""
    
    def __init__(self, model_path, conf_threshold=0.5, skip_frames=2, inference_width=640):
        """
        Initialize the GUI
        
        Args:
            model_path: Path to YOLOv8 model file
            conf_threshold: Confidence threshold for detections
            skip_frames: Process every nth frame (2 = process every 2nd frame)
            inference_width: Maximum width for inference (smaller = faster)
        """
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
        
        # For async detection
        self.current_frame = None
        self.current_detections = None
        self.detection_thread = None
        self.detection_lock = threading.Lock()
        self.detection_event = threading.Event()
        
        # Initialize detector
        try:
            self.detector = YOLOv8FaceDetector(model_path)
        except Exception as e:
            print(f"Error: {e}")
            return
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("YOLOv8 Face Detection - Live Camera (Optimized)")
        self.root.geometry("1024x768")
        
        # Info frame
        info_frame = tk.Frame(self.root, bg="lightgray")
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.info_label = Label(info_frame, text="Initializing...", bg="lightgray", fg="black", font=("Arial", 10))
        self.info_label.pack()
        
        # Video frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg="lightgray")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.status_label = Label(control_frame, text="Ready", bg="lightgray", fg="green", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = tk.Button(control_frame, text="Quit", command=self.on_quit, bg="red", fg="white")
        self.quit_button.pack(side=tk.RIGHT, padx=5)
        
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.config(text="Error: Could not open webcam", fg="red")
            return
        
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.update_frame()
    
    def update_frame(self):
        """Update frame from webcam with ultra-optimized rendering"""
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Process detection on every nth frame
        should_detect = (self.frame_count % self.skip_frames) == 0
        
        if should_detect:
            detections = self.detector.detect_faces_optimized(
                frame, 
                conf_threshold=self.conf_threshold,
                max_width=self.inference_width
            )
            self.current_detections = detections
            
            # Calculate detection FPS
            self.detection_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_detection_time
            if elapsed >= 1.0:
                self.detection_fps = self.detection_count / elapsed
                self.detection_count = 0
                self.last_detection_time = current_time
        else:
            detections = self.current_detections if self.current_detections else []
        
        # Draw detections directly on frame
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['confidence']
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_text = f"{conf:.1%}"
            cv2.putText(result, conf_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Calculate rendering FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
        
        # Add info text
        info_text = f"FPS: {self.fps:.1f} | Det: {self.detection_fps:.1f} | Faces: {len(detections)} | Skip: {self.skip_frames}x"
        cv2.putText(result, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize once for display
        h, w = result.shape[:2]
        display_w = 1000
        if w > display_w:
            display_h = int(h * display_w / w)
            result = cv2.resize(result, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Use numpy array directly instead of PIL for speed
        pil_image = Image.fromarray(rgb_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Update label
        self.video_label.imgtk = tk_image
        self.video_label.config(image=tk_image)
        
        # Update info (only every 30 frames to save CPU)
        if self.frame_count % 30 == 0:
            conf_str = ", ".join([f"{det['confidence']:.1%}" for det in detections[:3]])
            if len(detections) > 3:
                conf_str += "..."
            model_name = Path(self.model_path).stem
            info_text = f"{model_name} | Faces: {len(detections)} | [{conf_str}]"
            self.info_label.config(text=info_text)
        
        # Ultra-fast update loop
        self.root.after(1, self.update_frame)
    
    def on_quit(self):
        """Handle quit"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def webcam_detection_with_display(model_path, conf_threshold=0.5, skip_frames=2, inference_width=640):
    """
    Run face detection with live camera display using tkinter
    
    Args:
        model_path: Path to YOLOv8 model file
        conf_threshold: Confidence threshold (0.0-1.0)
        skip_frames: Process every nth frame (2 = skip 1, 3 = skip 2, etc.)
        inference_width: Maximum width for inference (smaller = faster)
    """
    gui = WebcamFaceDetectionGUI(model_path, conf_threshold, skip_frames, inference_width)
    gui.run()


def detect_from_image(image_path, model_path, output_path=None, conf_threshold=0.5):
    """
    Detect faces in a single image using YOLOv8
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLOv8 model file
        output_path: Path to save output image
        conf_threshold: Confidence threshold
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Initialize detector
    detector = YOLOv8FaceDetector(model_path)
    
    # Detect faces
    detections = detector.detect_faces(image, conf_threshold)
    print(f"Found {len(detections)} face(s)")
    
    for i, det in enumerate(detections, 1):
        print(f"  Face {i}: confidence = {det['confidence']:.2%}")
    
    # Draw detections
    result = detector.draw_faces(image, detections)
    
    # Save output
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Output saved to {output_path}")
    
    return result, detections


def detect_from_video(video_path, model_path, output_path=None, conf_threshold=0.5):
    """
    Detect faces in a video using YOLOv8
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLOv8 model file
        output_path: Path to save output video
        conf_threshold: Confidence threshold
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize detector
    detector = YOLOv8FaceDetector(model_path)
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect_faces(frame, conf_threshold)
        
        # Draw detections
        result = detector.draw_faces(frame, detections)
        
        # Add frame info
        cv2.putText(result, f"Frame: {frame_count}/{total_frames} | Faces: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        if writer:
            writer.write(result)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, found {len(detections)} face(s)")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
        print(f"Output video saved to {output_path}")
    
    print(f"Total frames processed: {frame_count}")


def detect_from_image(image_path, model_path, output_path=None, conf_threshold=0.5):
    """
    Detect faces in a single image using YOLOv8
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLOv8 model file
        output_path: Path to save output image
        conf_threshold: Confidence threshold
        
    Returns:
        Image with detected faces and detections list
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, []
    
    print(f"✓ Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    
    # Initialize detector
    detector = YOLOv8FaceDetector(model_path)
    
    # Detect faces
    print("Running face detection...")
    detections = detector.detect_faces(image, conf_threshold)
    
    # Draw detections
    result = detector.draw_faces(image.copy(), detections)
    
    # Add info text
    cv2.putText(result, f"Faces detected: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show detections info
    print(f"\n✓ Found {len(detections)} face(s)")
    for i, det in enumerate(detections, 1):
        print(f"  Face {i}: confidence={det['confidence']:.2%}, size={det['w']}x{det['h']}")
    
    # Save output if requested
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"\n✓ Output saved to {output_path}")
    
    return result, detections


def display_image_with_detections(image, title="Face Detection Result"):
    """
    Display image using tkinter window
    
    Args:
        image: OpenCV image (BGR)
        title: Window title
    """
    # Create window FIRST (required for PhotoImage)
    root = tk.Tk()
    root.title(title)
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if too large
    height, width = image_rgb.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200 / width, 800 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
    
    # Convert to PIL and tkinter (now that root exists)
    pil_image = Image.fromarray(image_rgb)
    photo = ImageTk.PhotoImage(pil_image)
    
    # Create label and display
    label = Label(root, image=photo)
    label.image = photo
    label.pack(padx=10, pady=10)
    
    print("✓ Image displayed. Close the window to continue...")
    root.mainloop()


def choose_and_process_file(script_dir):
    """
    Interactive menu to choose between webcam, video, or image
    Uses optimized models for each input type:
    - Webcam: yolov8n_100e (fastest, real-time)
    - Video: yolov8m_200e (balanced speed/accuracy)
    - Image: yolov8l_100e (best accuracy, static)
    
    Args:
        script_dir: Directory containing model files
    """
    while True:
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "  YOLOv8 Face Detection System - Choose Input Source".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        print("  1) WEBCAM        (Real-time detection with yolov8n - FASTEST)")
        print("  2) VIDEO FILE    (File processing with yolov8m - BALANCED)")
        print("  3) IMAGE FILE    (Static image with yolov8l - BEST ACCURACY)")
        print("  0) EXIT\n")
        print("="*70)
        
        choice = input("  Choose option (0-3): ").strip()
        
        if choice == "1":
            # Webcam - use fastest model
            print("\n" + "="*70)
            print("WEBCAM MODE - Using YOLOv8 Nano (Fastest)")
            print("="*70)
            model_path = script_dir / "yolov8n_100e.pt"
            
            if not model_path.exists():
                print(f"Error: Model not found at {model_path}")
                input("Press Enter to continue...")
                continue
            
            print(f"Model: {model_path.name}")
            print(f"Speed: Optimized for 30-60 FPS real-time detection")
            print(f"Close window to stop\n")
            
            try:
                webcam_detection_with_display(
                    model_path=str(model_path),
                    conf_threshold=0.35,
                    skip_frames=1,
                    inference_width=640
                )
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == "2":
            # Video - use balanced model
            print("\n" + "="*70)
            print("VIDEO MODE - Using YOLOv8 Medium (Balanced)")
            print("="*70)
            
            video_path = input("\nEnter path to video file: ").strip().strip('"\'')
            if not Path(video_path).exists():
                print(f"Error: Video file not found: {video_path}")
                input("Press Enter to continue...")
                continue
            
            model_path = script_dir / "yolov8m_200e.pt"
            if not model_path.exists():
                print(f"Error: Model not found at {model_path}")
                input("Press Enter to continue...")
                continue
            
            # Ask output options
            save_output = input("\nSave output video? (y/n): ").strip().lower() == 'y'
            output_path = None
            if save_output:
                output_path = input("Enter output path (default: output_video.mp4): ").strip().strip('"\'')
                if not output_path:
                    output_path = "output_video.mp4"
            
            print(f"\nModel: {model_path.name}")
            print(f"Input: {Path(video_path).name}")
            if output_path:
                print(f"Output: {output_path}")
            print(f"Processing...\n")
            
            try:
                detect_from_video(
                    video_path=video_path,
                    model_path=str(model_path),
                    output_path=output_path,
                    conf_threshold=0.35
                )
                print("Video processing complete!")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            # Image - use best accuracy model
            print("\n" + "="*70)
            print("IMAGE MODE - Using YOLOv8 Large (Best Accuracy)")
            print("="*70)
            
            image_path = input("\nEnter path to image file: ").strip().strip('"\'')
            if not Path(image_path).exists():
                print(f"Error: Image file not found: {image_path}")
                input("Press Enter to continue...")
                continue
            
            model_path = script_dir / "yolov8l_100e.pt"
            if not model_path.exists():
                print(f"Error: Model not found at {model_path}")
                input("Press Enter to continue...")
                continue
            
            # Ask output options
            save_output = input("\nSave output image? (y/n): ").strip().lower() == 'y'
            output_path = None
            if save_output:
                output_path = input("Enter output path (default: output_image.jpg): ").strip().strip('"\'')
                if not output_path:
                    output_path = "output_image.jpg"
            
            print(f"\nModel: {model_path.name}")
            print(f"Input: {Path(image_path).name}")
            if output_path:
                print(f"Output: {output_path}")
            print(f"Processing...\n")
            
            try:
                result, detections = detect_from_image(
                    image_path=image_path,
                    model_path=str(model_path),
                    output_path=output_path,
                    conf_threshold=0.35
                )
                
                if result is not None:
                    print(f"\nImage processing complete!")
                    display_result = input("\nDisplay result? (y/n): ").strip().lower() == 'y'
                    if display_result:
                        display_image_with_detections(result, title=f"Face Detection - {Path(image_path).name}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nPress Enter to continue...")
        
        elif choice == "0":
            print("\nExiting Face Detection System...")
            break
        
        else:
            print("\nInvalid choice. Please enter 0-3")
            input("Press Enter to continue...")


if __name__ == "__main__":
    print("\nInitializing YOLOv8 Face Detection System...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Check if required models exist
    required_models = [
        "yolov8n_100e.pt",
        "yolov8m_200e.pt",
        "yolov8l_100e.pt"
    ]
    
    missing_models = []
    for model in required_models:
        if not (script_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"\nWARNING: Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print(f"\nPlace these files in: {script_dir}")
        print("\nAvailable models:")
        available = [f.name for f in script_dir.glob("*.pt")]
        if available:
            for model in sorted(available):
                print(f"   {model}")
        input("\nPress Enter to continue anyway...")
    else:
        print("All models found!")
    
    print()
    
    try:
        choose_and_process_file(script_dir)
    except KeyboardInterrupt:
        print("\n\n⏸Interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
