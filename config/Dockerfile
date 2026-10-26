FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Đảm bảo quyền ghi cho các thư mục data
RUN mkdir -p data/uploads data/samples data/outputs && chmod -R 777 /app

EXPOSE 7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/web_app.py"]
