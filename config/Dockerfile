FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libturbojpeg0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs && chmod -R 777 /app

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.web_app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
