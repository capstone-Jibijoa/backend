#!/bin/bash
set -e

echo "Stopping existing docker container..."

CONTAINER_ID=$(sudo docker ps -q --filter "name=backend" 2>/dev/null || true)

if [ ! -z "$CONTAINER_ID" ]; then
    sudo docker stop backend
    sudo docker rm backend
    echo "Container stopped and removed."
else
    echo "No container to stop."
fi