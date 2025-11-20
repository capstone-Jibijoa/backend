#!/bin/bash
set -e

echo "Starting docker container..."

# 이전 컨테이너 있으면 중지 후 삭제
sudo docker stop backend 2>/dev/null || true
sudo docker rm backend 2>/dev/null || true

# 컨테이너 실행
# DB_PASSWORD는 컨테이너 내부에서 APP_SECRET_CONFIG_NAME을 통해 Secrets Manager에서 가져옴
sudo docker run -d \
  --name backend \
  -p 8000:8000 \
  --restart always \
  -e APP_SECRET_CONFIG_NAME="prod/backend/secrets" \
  -e DEFAULT_REGION="ap-southeast-2" \
  -e DB_HOST="project-main-db.crkcc42287ai.ap-southeast-2.rds.amazonaws.com" \
  -e DB_NAME="project_db" \
  -e DB_USER="backend_user" \
  -e PORT="5432" \
  -e QDRANT_HOST="52.63.128.220" \
  -e QDRANT_PORT="6333" \
  -e QDRANT_COLLECTION_WELCOME_NAME="welcome_subjective_vectors" \
  -e QDRANT_COLLECTION_QPOLL_NAME="qpoll_vectors_v2" \
  -e TRANSFORMERS_CACHE="/home/appuser/.cache/huggingface" \
  -e HF_HOME="/home/appuser/.cache/huggingface" \
  121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest

echo "Container started successfully."

# 컨테이너 상태 확인
sleep 3
sudo docker ps | grep backend

# 컨테이너 로그 확인
echo "Container logs:"
sudo docker logs backend --tail 20