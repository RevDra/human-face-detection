"""
Flask Web Application for YOLOv12 Face Detection
Supports image upload, video upload, and live webcam streaming
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import cv2
import numpy as np
import base64
import logging

from face_detection_yolov12 import YOLOv12FaceDetector, detect_from_video

# Initialize Flask app
app = Flask(__name__, template_folder='../web/templates')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "data" / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_MODELS = {
    'yolov12n-face.pt',
    'yolov12s-face.pt',
    'yolov12m-face.pt',
    'yolov12l-face.pt'
}
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Model cache
detector_cache = {}


def get_detector(model_name):
    """Get or create detector instance (cached)"""
    safe_name = secure_filename(model_name)

    if safe_name not in ALLOWED_MODELS:
        logging.error(f"Attempt to load unsupported model: {safe_name}")
        raise ValueError(f"Unsupported model: {safe_name}")

    if safe_name not in detector_cache:
        model_path = MODELS_DIR / safe_name

        try:
            final_path = model_path.resolve()
            safe_root = MODELS_DIR.resolve()
            if not str(final_path).startswith(str(safe_root)):
                logging.error(f"Security Alert: Symlink attack detected! {final_path}")
                raise ValueError("Invalid model path (Symlink violation)")
                
        except Exception as e:
            logging.error(f"Error resolving model path: {str(e)}")
            raise FileNotFoundError(f"Model path error: {str(e)}")

        if not final_path.exists():
            logging.error(f"Model file not found: {final_path}")
            raise FileNotFoundError(f"Model not found: {final_path}")

        detector_cache[safe_name] = YOLOv12FaceDetector(str(final_path))
        
    return detector_cache[safe_name]


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image(filename):
    """Check if file is image"""
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'jpg', 'jpeg', 'png', 'gif'}


def is_video(filename):
    """Check if file is video"""
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'mp4', 'avi', 'mov', 'mkv'}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    """Detect faces in uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename) or not is_image(file.filename):
            return jsonify({'error': 'Only image files allowed'}), 400
        
        # Get model selection
        model = request.form.get('model', 'yolov12l-face.pt')
        if model not in ALLOWED_MODELS:
            app.logger.info(f"Invalid model '{model}' requested. Fallback to default.")
            model = 'yolov12l-face.pt'
        
        # Get detector
        detector = get_detector(model)
        
        # Read image directly from file object
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # For webcam frames, use optimized detection with reduced resolution
        is_webcam = 'webcam' in file.filename.lower()
        if is_webcam:
            # Use optimized detection for speed
            detections = detector.detect_faces_optimized(image, conf_threshold=0.35, max_width=480)
        else:
            # Use standard detection for uploaded files
            detections = detector.detect_faces(image, conf_threshold=0.35)
        
        # Draw detections
        result_image = detector.draw_faces(image, detections, show_confidence=True)
        
        if result_image is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Convert result to base64 for display
        _, buffer = cv2.imencode('.jpg', result_image)
        img_base64 = base64.b64encode(buffer).decode()
        
        # Prepare response
        response = {
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': {
                'count': len(detections),
                'faces': [
                    {
                        'id': i + 1,
                        'confidence': f"{det['confidence']:.2%}",
                        'width': det['w'],
                        'height': det['h'],
                        'position': f"({det['x1']}, {det['y1']})"
                    }
                    for i, det in enumerate(detections)
                ]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.exception("Error during image detection")
        return jsonify({'error': 'Internal server error during image detection'}), 500


@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    """Detect faces in uploaded video"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename) or not is_video(file.filename):
            return jsonify({'error': 'Only video files allowed'}), 400
        
        # Get model selection
        model = request.form.get('model', 'yolov12m-face.pt')
        if model not in ALLOWED_MODELS:
            app.logger.info(f"Invalid model '{model}' requested. Fallback to default.")
            model = 'yolov12m-face.pt'
            
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = UPLOAD_FOLDER / f"input_{filename}"
        output_path = UPLOAD_FOLDER / f"output_{filename}"
        file.save(input_path)
        
        # Detect faces in video
        detect_from_video(
            video_path=str(input_path),
            model_path=str(MODELS_DIR / model),
            output_path=str(output_path),
            conf_threshold=0.35
        )
        
        # Return file info
        response = {
            'success': True,
            'message': 'Video processing complete',
            'output_file': output_path.name,
            'download_url': f'/api/download/{output_path.name}'
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Log the full exception server-side without exposing details to the client
        app.logger.exception("Error while processing video detection request")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed file"""
    try:
        filepath = UPLOAD_FOLDER / secure_filename(filename)
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        # Log the full exception server-side without exposing details to the client
        app.logger.exception("Error while processing download request for %s", filename)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get ALL available models for dropdown selection"""
    # Cập nhật danh sách đầy đủ 4 models
    models = {
        'nano': {
            'name': 'yolov12n-face.pt',
            'label': 'Nano (n) - Fastest',
            'description': 'Real-time speed, best for CPU/Webcam',
            'size': 'Smallest'
        },
        'small': {
            'name': 'yolov12s-face.pt',
            'label': 'Small (s) - Balanced',
            'description': 'Good balance of speed and accuracy',
            'size': 'Small'
        },
        'medium': {
            'name': 'yolov12m-face.pt',
            'label': 'Medium (m) - High Precision',
            'description': 'High accuracy, requires decent GPU',
            'size': 'Medium'
        },
        'large': {
            'name': 'yolov12l-face.pt',
            'label': 'Large (l) - Max Accuracy',
            'description': 'Best detection quality, slowest speed',
            'size': 'Large'
        }
    }
    
    # Chỉ trả về những model thực sự tồn tại trong thư mục
    available = {}
    for key, info in models.items():
        model_path = MODELS_DIR / info['name']
        if model_path.exists():
            available[key] = info
    
    # Sắp xếp theo thứ tự kích thước để hiển thị đẹp hơn
    order = ['nano', 'small', 'medium', 'large']
    sorted_available = {k: available[k] for k in order if k in available}
    
    return jsonify(sorted_available)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Face Detection API'})


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded"""
    return jsonify({'error': 'File too large. Maximum 500MB allowed'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 Starting YOLOv12 Face Detection Web Server")
    print("="*70)
    print("\n📍 Server: http://localhost:5000")
    print("📁 Upload folder: ", UPLOAD_FOLDER)
    print("🔧 Models folder: ", MODELS_DIR)
    print("\n🎯 Available endpoints:")
    print("   GET  /                    - Web interface")
    print("   POST /api/detect-image    - Detect faces in image")
    print("   POST /api/detect-video    - Detect faces in video")
    print("   GET  /api/models          - Get available models")
    print("   GET  /api/health          - Health check")
    print("\n" + "="*70 + "\n")

    # Determine debug mode from environment (default: disabled)
    debug_mode = os.getenv("FLASK_ENV") == "development"
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=7860,
        debug=debug_mode,
        use_reloader=False
    )
