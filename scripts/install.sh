#!/bin/bash
set -e

echo "Pulling latest docker image from ECR..."

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
echo "Logging in to ECR..."
aws ecr get-login-password --region ap-southeast-2 \
    | sudo docker login --username AWS --password-stdin 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com

# 이전 이미지 제거 (디스크 공간 확보)
echo "Removing old images..."
sudo docker images 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend -q | xargs -r sudo docker rmi -f || true

# 최신 이미지 pull
echo "Pulling latest image..."
sudo docker pull 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest

echo "Image pulled successfully."