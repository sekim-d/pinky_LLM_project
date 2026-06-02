#!/bin/bash
set -e

echo "[1] Ollama에 파인튜닝 모델 등록 중..."
ollama create pinky-warehouse -f ./Modelfile

echo "[2] 등록 확인..."
ollama list | grep pinky-warehouse

echo ""
echo "완료! 이제 docker-compose.yml 에서 MODEL 환경변수를 수정하세요:"
echo "  MODEL=pinky-warehouse"
