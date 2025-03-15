#!/bin/bash
set -e  # Exit on error

echo "Setting up YOLOv12 environment using Python venv..."

# Use available Python version
PYTHON_CMD=python3
echo "Using $(python3 --version)"

# Check disk space
echo "Checking available disk space..."
df -h .

# Download flash-attention wheel file if it doesn't exist
if [ ! -f flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl ]; then
    echo "Downloading flash-attention wheel file..."
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
fi

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating Python virtual environment 'env'..."
    $PYTHON_CMD -m venv env
else
    echo "Virtual environment 'env' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements with pip cache
echo "Installing requirements..."
pip install --no-cache-dir -r requirements.txt

# Note about flash-attention
echo "Note: The flash-attention wheel file is for Python 3.11 and CUDA 11, but we're using Python $(python --version | cut -d' ' -f2)."
echo "We'll skip installing it directly, but you can try installing it manually if needed."

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo ""
echo "Installation complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source env/bin/activate"
echo ""
echo "After activation, you can run the training with:"
echo "python src/train_yolov12.py"
echo ""
echo "And validation with:"
echo "python src/validate_yolov12.py" 