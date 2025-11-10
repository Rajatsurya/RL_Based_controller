# TD3 RL Controller for Robot Navigation

This package implements a TD3 (Twin Delayed Deep Deterministic Policy Gradient) reinforcement learning agent for robot navigation. The agent takes a 30-dimensional state vector and outputs linear and angular velocities for robot control.

## Components

### 1. State Space Vector Builder (`statespace_vector.py`)
- Processes sensor data (LIDAR, IMU, odometry, goals)
- Creates a 30-dimensional state vector
- Publishes state vector on `/rl_state` topic

### 2. TD3 Agent (`td3_agent.py`)
- Implements TD3 algorithm with actor-critic networks
- Takes 30-dimensional state input
- Outputs 2-dimensional action (linear.x, angular.z)
- Publishes actions on `/cmd_vel` topic

### 3. Reward Function (`reward_function.py`)
- Calculates rewards based on navigation performance
- Considers distance to goal, angle alignment, velocity
- Publishes rewards on `/rl_reward` topic

### 4. Training Launcher (`training_launcher.py`)
- Coordinates training process
- Tracks episode statistics
- Saves training progress

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make scripts executable:
```bash
chmod +x src/td3_rl_controller/src/*.py
```

## Usage

### Training Mode

1. Start the training system:
```bash
roslaunch td3_rl_controller td3_training.launch
```

2. The system will automatically:
   - Build state vectors from sensor data
   - Calculate rewards based on navigation performance
   - Train the TD3 agent
   - Save models periodically

### Evaluation Mode

1. Load a trained model:
```bash
roslaunch td3_rl_controller td3_evaluation.launch model_path:=/path/to/trained_model.pth
```

## State Vector (30 dimensions)

The state vector consists of:
- **LIDAR data (21)**: Downsampled laser scan readings
- **IMU data (3)**: Roll, pitch, yaw
- **Velocity data (2)**: Linear.x, angular.z
- **Goal relative (4)**: dx, dy, dyaw, distance

## Action Space (2 dimensions)

- **Linear velocity**: Range [-1.0, 1.0] m/s
- **Angular velocity**: Range [-1.0, 1.0] rad/s

## Reward Function

The reward function considers:
- **Distance reward**: Positive for moving toward goal
- **Angle reward**: Positive for aligning with goal direction
- **Velocity reward**: Small positive for forward motion
- **Goal reached**: Large positive reward (+100)
- **Time penalty**: Small negative per step (-0.01)

## Training Parameters

- **Episodes**: 1000 (configurable)
- **Episode timeout**: 300 seconds
- **Batch size**: 100
- **Learning rate**: 3e-4
- **Discount factor**: 0.99
- **Target update rate**: 0.005

## Model Architecture

### Actor Network
- Input: 30 dimensions
- Hidden layers: 400 → 300 → 2
- Activation: ReLU → ReLU → Tanh
- Output: Linear and angular velocities

### Critic Network
- Input: 30 (state) + 2 (action) = 32 dimensions
- Two Q-networks for TD3
- Hidden layers: 400 → 300 → 1
- Activation: ReLU → ReLU → Linear

## Files

- `statespace_vector.py`: State vector builder
- `td3_agent.py`: TD3 agent implementation
- `reward_function.py`: Reward calculation
- `training_launcher.py`: Training coordination
- `td3_training.launch`: Launch file for training
- `requirements.txt`: Python dependencies

## Topics

### Subscribed Topics
- `/scan`: Laser scan data
- `/imu`: IMU data
- `/odom`: Odometry data
- `/move_base/goal`: Navigation goals

### Published Topics
- `/rl_state`: State vector (Float32MultiArray)
- `/rl_reward`: Reward and done flag (Float32MultiArray)
- `/cmd_vel`: Robot velocity commands (Twist)

## Training Progress

Training statistics are saved to `/tmp/td3_training_stats.pkl` and include:
- Episode rewards
- Episode lengths
- Best reward achieved
- Current episode number

## Tips for Training

1. **Start with simple environments**: Begin training in open spaces
2. **Monitor rewards**: Watch for increasing average rewards
3. **Adjust hyperparameters**: Modify reward weights if needed
4. **Use visualization**: Enable RViz to see robot behavior
5. **Save frequently**: Models are saved every 100 episodes

## Troubleshooting

### Common Issues

1. **No sensor data**: Check if robot sensors are publishing
2. **No movement**: Verify `/cmd_vel` topic is being subscribed
3. **Poor performance**: Adjust reward function weights
4. **Memory issues**: Reduce batch size or replay buffer size

### Debug Commands

```bash
# Check topics
rostopic list
rostopic echo /rl_state
rostopic echo /rl_reward

# Check node status
rosnode list
rosnode info /td3_agent

# Monitor training
tail -f /tmp/td3_training_stats.pkl
```

## Customization

### Modifying State Vector
Edit `statespace_vector.py` to change state representation.

### Adjusting Rewards
Modify `reward_function.py` to change reward calculation.

### Changing Network Architecture
Update `td3_agent.py` to modify actor/critic networks.

### Adding New Sensors
Extend state vector in `statespace_vector.py` and update dimensions. 