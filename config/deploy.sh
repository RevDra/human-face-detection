#!/bin/bash
# --- FIX PATH FOR FOLDER CONFIG ---
cd "$(dirname "$0")/.."
echo "[INFO] Working Directory set to: $(pwd)"
# ----------------------------------
# YOLOv12 Face Detection - Production Deployment Script
# Usage: bash deploy.sh [start|stop|restart|status|logs|deploy]

set -e

APP_NAME="face-detection-yolov12"
APP_USER="facedetection"
APP_HOME="/opt/face-detection"
LOG_DIR="/var/log/face-detection"
SERVICE_NAME="face-detection"
PORT="7860"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo_error "This script must be run as root (sudo)"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    echo_info "Installing system dependencies..."
    apt-get update
    # Thêm libgl1 cho OpenCV
    apt-get install -y python3.10 python3-pip python3-venv
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1
    apt-get install -y nginx supervisor
    echo_info "Dependencies installed successfully"
}

# Setup application
setup_app() {
    echo_info "Setting up application..."
    
    # Create app directory
    mkdir -p $APP_HOME
    
    # Create app user
    if ! id "$APP_USER" &>/dev/null; then
        useradd -r -s /bin/bash -d $APP_HOME $APP_USER
        echo_info "Created app user: $APP_USER"
    fi
    
    # Create virtual environment
    if [ ! -d "$APP_HOME/venv" ]; then
        python3 -m venv $APP_HOME/venv
        echo_info "Created virtual environment"
    fi
    
    # Create log directory
    mkdir -p $LOG_DIR
    chown $APP_USER:$APP_USER $LOG_DIR
    chmod 755 $LOG_DIR

    # Create uploads directory
    mkdir -p "$APP_HOME/data/uploads"
    mkdir -p "$APP_HOME/models"
    
    # Authorized user app
    chown -R $APP_USER:$APP_USER $APP_HOME
    chmod -R 777 "$APP_HOME/data"
}

# Check and setup models
check_models() {
    echo_info "Checking YOLOv12 models..."
    MISSING=0
    MODELS=("yolov12n-face.pt" "yolov12s-face.pt" "yolov12m-face.pt" "yolov12l-face.pt")
    
    for model in "${MODELS[@]}"; do
        if [ ! -f "$APP_HOME/models/$model" ]; then
            # Kiểm tra xem có file ở thư mục hiện tại không để copy vào
            if [ -f "./models/$model" ]; then
                cp "./models/$model" "$APP_HOME/models/"
                echo_info "Copied $model to installation dir"
            else
                echo_warning "Missing model: $model"
                MISSING=1
            fi
        fi
    done
    
    if [ $MISSING -eq 1 ]; then
        echo_warning "Some YOLOv12 models are missing in $APP_HOME/models/"
    fi
    
    chown -R $APP_USER:$APP_USER "$APP_HOME/models"
}

# Install Python dependencies
install_python_deps() {
    echo_info "Installing Python dependencies..."
    source $APP_HOME/venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    echo_info "Python dependencies installed"
}

# Setup Gunicorn with Supervisor
setup_supervisor() {
    echo_info "Setting up Supervisor..."
    
    cat > /etc/supervisor/conf.d/face-detection.conf <<EOF
[program:face-detection]
# Lưu ý: trỏ vào src.web_app:app vì file nằm trong folder src
command=$APP_HOME/venv/bin/gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 src.web_app:app
directory=$APP_HOME
user=$APP_USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$LOG_DIR/gunicorn.log
environment=PATH=$APP_HOME/venv/bin,PYTHONUNBUFFERED=1,PYTHONPATH=$APP_HOME
EOF
    
    supervisorctl reread
    supervisorctl update
    echo_info "Supervisor configured"
}

# Setup Nginx reverse proxy
setup_nginx() {
    echo_info "Setting up Nginx reverse proxy..."
    
    cat > /etc/nginx/sites-available/face-detection <<EOF
upstream face_detection {
    server 0.0.0.0:$PORT fail_timeout=0;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 500M;
    
    access_log /var/log/nginx/face-detection-access.log;
    error_log /var/log/nginx/face-detection-error.log;
    
    location / {
        proxy_pass http://face_detection;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_redirect off;
        
        # Timeouts for large file uploads
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
    
    location /api/download {
        proxy_pass http://face_detection;
        proxy_set_header Host \$host;
        proxy_buffering off;
    }
}
EOF
    
    ln -sf /etc/nginx/sites-available/face-detection /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    nginx -t
    systemctl restart nginx
    echo_info "Nginx configured"
}

# Start application
start_app() {
    echo_info "Starting application..."
    supervisorctl start face-detection
    sleep 2
    
    if supervisorctl status face-detection | grep -q "RUNNING"; then
        echo_info "Application started successfully"
        echo_info "Internal App Port: $PORT"
        echo_info "Public Access: http://localhost (via Nginx)"
    else
        echo_error "Failed to start application"
        supervisorctl tail face-detection
        exit 1
    fi
}

# Stop application
stop_app() {
    echo_info "Stopping application..."
    supervisorctl stop face-detection
    echo_info "Application stopped"
}

# Restart application
restart_app() {
    echo_info "Restarting application..."
    supervisorctl restart face-detection
    sleep 2
    
    if supervisorctl status face-detection | grep -q "RUNNING"; then
        echo_info "Application restarted successfully"
    else
        echo_error "Failed to restart application"
        supervisorctl tail face-detection
        exit 1
    fi
}

# Check status
check_status() {
    echo_info "Application Status:"
    supervisorctl status face-detection
    
    echo -e "\n${GREEN}System Status:${NC}"
    systemctl status nginx --no-pager || true
}

# Show logs
show_logs() {
    echo_info "Application Logs:"
    tail -f $LOG_DIR/gunicorn.log
}

# Full deployment
full_deploy() {
    echo_info "Starting full deployment (YOLOv12 Production)..."
    check_root
    install_deps
    setup_app
    
    # Copy application files
    echo_info "Copying application files..."
    # Copy all except venv & git
    rsync -av --progress . $APP_HOME --exclude venv --exclude .git
    chown -R $APP_USER:$APP_USER $APP_HOME
    
    check_models
    install_python_deps
    setup_supervisor
    setup_nginx
    start_app
    
    echo_info "Deployment completed successfully!"
    echo ""
    echo "=========================================="
    echo "YOLOv12 Face Detection Deployed!"
    echo "=========================================="
    echo "Public URL: http://localhost (Port 80 -> $PORT)"
    echo "Logs: tail -f $LOG_DIR/gunicorn.log"
    echo "Status: supervisorctl status face-detection"
    echo ""
}

# Parse command line arguments
case "${1:-status}" in
    install)
        check_root
        install_deps
        ;;
    setup)
        check_root
        setup_app
        ;;
    start)
        check_root
        start_app
        ;;
    stop)
        check_root
        stop_app
        ;;
    restart)
        check_root
        restart_app
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    deploy)
        full_deploy
        ;;
    *)
        echo "Usage: $0 {install|setup|start|stop|restart|status|logs|deploy}"
        exit 1
        ;;
esac
