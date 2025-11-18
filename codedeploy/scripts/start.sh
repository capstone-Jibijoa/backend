#!/bin/bash
echo "Starting docker container..."

docker run -d \
  --name backend \
  -p 8000:8000 \
  --env-file /home/ec2-user/backend/.env \
  121568407787.dkr.ecr.ap-southeast-2.amazonaws.com/backend-api:latest