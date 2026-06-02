FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY camera_server.py .

RUN pip install --no-cache-dir opencv-python-headless

ENV PYTHONUNBUFFERED=1
CMD ["python3", "camera_server.py"]
