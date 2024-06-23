#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

deploy_success=false

# Check if Docker Compose is available
if command -v docker compose &> /dev/null; then
    echo "Deploying with Docker Compose..."
    if docker compose up --build -d; then
        deploy_success=true
    fi
else
    echo "Docker Compose not found. Deploying with pure Docker..."
    
    # Build Docker image
    if docker build -t iris-classifier .; then
        # Run Docker container
        if docker run -d -p 8000:8000 -v $(pwd)/src:/app -v $(pwd)/models:/app/models iris-classifier; then
            deploy_success=true
        fi
    fi
fi

# Check if deployment was successful
if [ "$deploy_success" = true ]; then
    echo "Deployment complete. API is available at http://<your-ip>:8000"
else
    echo "Deployment failed. Please check the error messages above."
    exit 1
fi
