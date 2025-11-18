#!/bin/bash
echo "Starting docker container..."

# 이전 컨테이너 있으면 중지 후 삭제
docker stop backend || true
docker rm backend || true

docker run -d \
  --name backend \
  -p 8000:8000 \
  121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest