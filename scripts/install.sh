#!/bin/bash
set -e

echo "Pulling docker image..."

# IMAGE_TAG 환경변수가 설정되어 있는지 확인
if [ -z "$IMAGE_TAG" ]; then
    echo "ERROR: IMAGE_TAG environment variable is not set"
    exit 1
fi

echo "IMAGE_TAG: $IMAGE_TAG"

# jq 설치 확인 및 설치
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Installing jq..."
    sudo yum install -y jq
fi

# Docker가 설치되어 있는지 확인
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
fi

# ECR 로그인
aws ecr get-login-password --region ap-southeast-2 \
    | sudo docker login --username AWS --password-stdin 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com

# 이미지 pull
sudo docker pull 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:${IMAGE_TAG}

echo "Image pulled successfully."