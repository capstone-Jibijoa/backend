#!/bin/bash
echo "Stopping existing docker container..."

CONTAINER_ID=$(docker ps -q --filter "name=backend")

if [ ! -z "$CONTAINER_ID" ]; then
    docker stop backend
    docker rm backend
    echo "Container stopped and removed."
else
    echo "No container to stop."
fi