#!/bin/bash

# Navigate to the python-server directory
cd ../python-server

# Upgrade pip to the latest version
pip install --upgrade pip

# Create Python virtual environment for the server if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install necessary Python packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

# Create TensorFlow test environment if it doesn't exist
if [ ! -d "tf-test-env" ]; then
    python -m venv tf-test-env
fi

# Activate the TensorFlow test environment
source tf-test-env/bin/activate

# Install TensorFlow (or other required packages)
pip install tensorflow

# Deactivate the TensorFlow test environment
deactivate

# Start the Flask server
source venv/bin/activate
python -m flask run
