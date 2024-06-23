#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Variables
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements-lab3.txt"

# Function to create and activate virtual environment
create_and_activate_venv() {
    python3 -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
}

# Function to install dependencies
install_dependencies() {
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
}

# Function to run Python scripts
run_python_scripts() {
    python src/create_dataset.py
    python src/train_model.py
}

# Function to deploy using Docker Compose
deploy_with_docker_compose() {
    echo "Deploying with Docker Compose..."
    docker compose up --build -d
}

# Function to deploy using Docker
deploy_with_docker() {
    echo "Docker Compose not found. Deploying with pure Docker..."

    # Build Docker image
    docker build -t iris-classifier .

    # Run Docker container
    docker run -d -p 8000:8000 -v "$(pwd)/src:/app" -v "$(pwd)/models:/app/models" iris-classifier
}

# Main script execution
echo "Creating and activating Python virtual environment..."
create_and_activate_venv

echo "Installing dependencies..."
install_dependencies

echo "Running Python scripts to create dataset and train model..."
run_python_scripts

deploy_success=false

# Check if Docker Compose is available
if command -v docker compose &> /dev/null; then
    if deploy_with_docker_compose; then
        deploy_success=true
    fi
else
    if deploy_with_docker; then
        deploy_success=true
    fi
fi

# Check if deployment was successful
if [ "$deploy_success" = true ]; then
    echo "Deployment complete. API is available at http://<your-ip>:8000"
else
    echo "Deployment failed. Please check the error messages above."
    exit 1
fi
