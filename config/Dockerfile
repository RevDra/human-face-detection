FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/samples data/outputs

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/web_app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/web_app.py"]
