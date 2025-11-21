#!/bin/bash
set -e

echo "Starting docker container..."

# 스크립트 디렉토리 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# IMAGE_TAG 파일에서 읽기
if [ -f "$SCRIPT_DIR/IMAGE_TAG.txt" ]; then
    IMAGE_TAG=$(cat "$SCRIPT_DIR/IMAGE_TAG.txt")
    echo "Loaded IMAGE_TAG from file: $IMAGE_TAG"
else
    echo "ERROR: IMAGE_TAG.txt file not found in $SCRIPT_DIR"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

# IMAGE_TAG 확인
if [ -z "$IMAGE_TAG" ]; then
    echo "ERROR: IMAGE_TAG is empty"
    exit 1
fi

echo "IMAGE_TAG: $IMAGE_TAG"

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