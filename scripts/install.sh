#!/bin/bash
set -e

echo "Pulling docker image..."

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