#!/bin/bash
set -e

echo "Pulling latest docker image..."

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

# 최신 이미지 pull
sudo docker pull 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest

echo "Image pulled successfully."