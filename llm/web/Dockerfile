FROM python:3.12-slim

WORKDIR /app
COPY app.py index.html ./

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" requests pydantic

CMD ["python3", "app.py"]
