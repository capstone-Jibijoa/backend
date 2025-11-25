#!/bin/bash
set -e

echo "Pulling docker image..."

# ECR 로그인
aws ecr get-login-password --region ap-southeast-2 \
    | sudo docker login --username AWS --password-stdin 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com

sudo docker rmi 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest --force 2>/dev/null || true

# 이미지 pull
sudo docker pull 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest

echo "Image pulled successfully."