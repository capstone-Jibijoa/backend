#!/bin/bash
echo "Pulling latest docker image..."

aws ecr get-login-password --region ap-southeast-2 \
    | docker login --username AWS --password-stdin 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com

docker pull 121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/jibijoa-backend:latest
