#!/bin/bash
# YOLOv8 Face Detection - Production Deployment Script
# Usage: bash deploy.sh [start|stop|restart|status|logs]

set -e

APP_NAME="face-detection-web"
APP_USER="facedetection"
APP_HOME="/opt/face-detection"
LOG_DIR="/var/log/face-detection"
SERVICE_NAME="face-detection"
PORT="5000"

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
        echo_error "This script must be run as root"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    echo_info "Installing system dependencies..."
    apt-get update
    apt-get install -y python3.10 python3-pip python3-venv
    apt-get install -y libsm6 libxext6 libxrender-dev
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
    python3.10 -m venv $APP_HOME/venv
    echo_info "Created virtual environment"
    
    # Create log directory
    mkdir -p $LOG_DIR
    chown $APP_USER:$APP_USER $LOG_DIR
    chmod 755 $LOG_DIR
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
command=$APP_HOME/venv/bin/gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 web_app:app
directory=$APP_HOME
user=$APP_USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$LOG_DIR/gunicorn.log
environment=PATH=$APP_HOME/venv/bin,PYTHONUNBUFFERED=1
EOF
    
    supervisorctl reread
    supervisorctl update
    echo_info "Supervisor configured"
}

# Setup Nginx reverse proxy
setup_nginx() {
    echo_info "Setting up Nginx reverse proxy..."
    
    cat > /etc/nginx/sites-available/face-detection <<'EOF'
upstream face_detection {
    server 0.0.0.0:5000 fail_timeout=0;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 500M;
    
    access_log /var/log/nginx/face-detection-access.log;
    error_log /var/log/nginx/face-detection-error.log;
    
    # Redirect HTTP to HTTPS (uncomment after setting up SSL)
    # return 301 https://$server_name$request_uri;
    
    location / {
        proxy_pass http://face_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Timeouts for large file uploads
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
    
    location /api/download {
        proxy_pass http://face_detection;
        proxy_set_header Host $host;
        proxy_buffering off;
    }
}

# HTTPS configuration (uncomment after setting up SSL with Let's Encrypt)
# server {
#     listen 443 ssl http2;
#     server_name YOUR_DOMAIN;
#     
#     ssl_certificate /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem;
#     
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers HIGH:!aNULL:!MD5;
#     ssl_prefer_server_ciphers on;
#     
#     client_max_body_size 500M;
#     
#     location / {
#         proxy_pass http://face_detection;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto https;
#     }
# }
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
        echo_info "Access at http://localhost:$PORT"
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
    echo_info "Starting full deployment..."
    check_root
    install_deps
    setup_app
    
    # Copy application files
    echo_info "Copying application files..."
    cp -r . $APP_HOME/
    chown -R $APP_USER:$APP_USER $APP_HOME
    
    install_python_deps
    setup_supervisor
    setup_nginx
    start_app
    
    echo_info "Deployment completed successfully!"
    echo ""
    echo "=========================================="
    echo "Application deployed!"
    echo "=========================================="
    echo "URL: http://localhost"
    echo "Logs: tail -f $LOG_DIR/gunicorn.log"
    echo "Status: supervisorctl status face-detection"
    echo ""
    echo "Next steps:"
    echo "1. Verify app is running: curl http://localhost"
    echo "2. Setup SSL/HTTPS (see comments in Nginx config)"
    echo "3. Configure domain name"
    echo "4. Setup monitoring and alerts"
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
        echo ""
        echo "Commands:"
        echo "  install   - Install system dependencies"
        echo "  setup     - Setup application directory and user"
        echo "  start     - Start the application"
        echo "  stop      - Stop the application"
        echo "  restart   - Restart the application"
        echo "  status    - Check application status"
        echo "  logs      - Follow application logs"
        echo "  deploy    - Full deployment (requires root)"
        exit 1
        ;;
esac
