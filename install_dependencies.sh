#!/bin/bash

echo "Installing TD3 RL Controller Dependencies"
echo "=========================================="

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install Python3 and pip3 first."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch numpy

# Check if ROS is available
if ! command -v roscore &> /dev/null; then
    echo "Warning: ROS not found. Make sure ROS is installed and sourced."
    echo "You can source ROS with: source /opt/ros/noetic/setup.bash"
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x src/*.py

echo "Installation complete!"
echo ""
echo "To test the installation, run:"
echo "  python3 src/test_td3.py"
echo ""
echo "To start training, run:"
echo "  roslaunch td3_rl_controller td3_training.launch" 