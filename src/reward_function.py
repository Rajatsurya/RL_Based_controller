#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Twist, PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32MultiArray, Bool
import math

class RewardFunction:
    def __init__(self):
        rospy.init_node("reward_function")
        
        # Goal tracking
        self.goal_position = np.array([0.0, 0.0])
        self.goal_yaw = 0.0
        self.current_position = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.current_velocity = np.array([0.0, 0.0])
        
        # Previous state for reward calculation
        self.prev_distance_to_goal = float('inf')
        self.prev_angle_to_goal = 0.0
        self.prev_velocity = np.array([0.0, 0.0])
        
        # Reward parameters (fixed defaults)
        self.distance_threshold = 0.1  # meters
        self.angle_threshold = 0.1     # radians
        self.max_velocity = 1.0        # m/s
        self.max_angular_velocity = 1.0 # rad/s
        
        # Reward weights (fixed defaults)
        self.w_distance = 1.0
        self.w_angle = 0.5
        self.w_velocity = 0.1
        self.w_collision = -1000.0
        self.w_goal_reached = 100.0
        self.w_time_penalty = -0.01
        
        # Collision detection parameters
        self.collision_threshold = rospy.get_param('~collision_threshold', 0.15)  # 15cm collision threshold
        self.collision_grace_steps = rospy.get_param('~collision_grace_steps', 10)  # steps to ignore collisions after episode start
        self.lidar_data = np.zeros(21)  # Store LiDAR data for collision detection
        
        # Testing parameters
        self.force_collision_test = rospy.get_param('~force_collision_test', False)  # Enable forced collision testing
        self.collision_test_step = rospy.get_param('~collision_test_step', 200)  # Step to force collision
        self.collision_test_interval = rospy.get_param('~collision_test_interval', 0)  # 0 = only once, >0 = every N episodes
        
        # Read topics from params
        odom_topic = rospy.get_param('~odom_topic', '/odom')
        goal_topic = rospy.get_param('~goal_topic', '/move_base/goal')
        simple_goal_topic = rospy.get_param('~simple_goal_topic', '/move_base_simple/goal')
        lidar_topic = rospy.get_param('~lidar_topic', '/scan')
        
        # Subscribers
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber(goal_topic, MoveBaseActionGoal, self.goal_callback)
        rospy.Subscriber(simple_goal_topic, PoseStamped, self.simple_goal_callback)
        rospy.Subscriber(lidar_topic, LaserScan, self.lidar_callback)
        
        # Publishers
        self.reward_pub = rospy.Publisher('/rl_reward', Float32MultiArray, queue_size=1)
        
        # Episode tracking
        self.episode_start_time = rospy.Time.now()
        self.max_episode_time = rospy.Duration(60)  # 1 minute
        self.episode_steps = 0
        self.max_episode_steps = 1000
        self.episode_return = 0.0
        self.goal_reached_last_step = False
        self.episode_count = 0  # Track episode number for testing
        # External episode control
        self.episode_active = True
        self.collision_latched = False
        rospy.Subscriber('/rl_episode_active', Bool, self.episode_active_callback)
        
        rospy.loginfo("Reward Function initialized")
    
    def odom_callback(self, msg):
        """Callback for odometry data"""
        # Ignore updates when episode is inactive
        if not self.episode_active:
            return
        # Update current position and orientation
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.current_position = np.array([pos.x, pos.y])
        (_, _, self.current_yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        
        # Update velocity
        twist = msg.twist.twist
        self.current_velocity = np.array([twist.linear.x, twist.angular.z])
        
        # Calculate and publish reward
        reward = self.calculate_reward()
        self.publish_reward(reward)
        
        # Update previous values
        self.prev_distance_to_goal = self.get_distance_to_goal()
        self.prev_angle_to_goal = self.get_angle_to_goal()
        self.prev_velocity = self.current_velocity.copy()
        
        self.episode_steps += 1

    def episode_active_callback(self, msg: Bool):
        """Trainer toggles episode activity; clear collision latch on activation."""
        was_active = self.episode_active
        self.episode_active = bool(msg.data)
        if self.episode_active and not was_active:
            # New episode is starting; clear latched collision
            self.collision_latched = False
    
    def goal_callback(self, msg: MoveBaseActionGoal):
        """Callback for goal updates"""
        pos = msg.goal.target_pose.pose.position
        ori = msg.goal.target_pose.pose.orientation
        self.goal_position = np.array([pos.x, pos.y])
        (_, _, self.goal_yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        rospy.loginfo(f"New goal set: {self.goal_position}, yaw: {self.goal_yaw}")
    
    def simple_goal_callback(self, msg: PoseStamped):
        """Callback for simple goal updates (PoseStamped)"""
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.goal_position = np.array([pos.x, pos.y])
        (_, _, self.goal_yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        rospy.loginfo(f"New goal set: {self.goal_position}, yaw: {self.goal_yaw}")
    
    def lidar_callback(self, msg: LaserScan):
        """Callback for LiDAR data - used for collision detection"""
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        ranges[np.isinf(ranges)] = msg.range_max
        
        # Downsample to 21 points (same as state space)
        step = len(ranges) // 21
        self.lidar_data = ranges[::step][:21]
        
        # Debug: Log LiDAR data periodically (commented out for cleaner logs)
        # if hasattr(self, 'episode_steps') and self.episode_steps % 100 == 0:
        #     min_dist = np.min(self.lidar_data)
        #     rospy.loginfo(f"LiDAR Debug - Min: {min_dist:.3f}m, Range: {msg.range_min:.2f}-{msg.range_max:.2f}m, Points: {len(ranges)}")
        #     rospy.loginfo(f"Downsampled LiDAR: {self.lidar_data[:5]} ... {self.lidar_data[-5:]}")
    
    def get_distance_to_goal(self):
        """Calculate distance to goal"""
        return np.linalg.norm(self.current_position - self.goal_position)
    
    def get_angle_to_goal(self):
        """Calculate angle to goal"""
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_yaw
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return abs(angle_diff)
    
    def check_collision(self):
        """Check for collision using LiDAR data"""
        # Check if any LiDAR reading is below collision threshold
        min_distance = np.min(self.lidar_data)
        
        # Debug: log minimum distance periodically (commented out for cleaner logs)
        # if self.episode_steps % 50 == 0:  # Every 50 steps
        #     rospy.loginfo(f"Min LiDAR distance: {min_distance:.3f}m (threshold: {self.collision_threshold:.3f}m)")
        #     print(f"DEBUG: Min LiDAR distance: {min_distance:.3f}m (threshold: {self.collision_threshold:.3f}m)")
        
        # Debug: Force collision detection for testing
        if self.force_collision_test:
            should_test = False
            
            # Check if we should test collision this episode
            if self.collision_test_interval == 0:
                # Test only once at specified step
                should_test = (self.episode_steps == self.collision_test_step)
            else:
                # Test every N episodes at specified step
                should_test = (self.episode_count % self.collision_test_interval == 0 and 
                             self.episode_steps == self.collision_test_step)
            
            if should_test:
                print(f"🧪 FORCED COLLISION TEST at episode {self.episode_count + 1}, step {self.episode_steps}")
                rospy.logerr("FORCED COLLISION TEST - Episode should terminate!")
                return True
        
        # Skip collision checks for first N steps after (re)start
        if self.episode_steps < self.collision_grace_steps:
            return False

        if (self.episode_active and not self.collision_latched and
            min_distance < self.collision_threshold):
            # Latch so we only terminate once per episode
            self.collision_latched = True
            print(f"🚨 COLLISION DETECTED! Minimum distance: {min_distance:.3f}m (threshold: {self.collision_threshold:.3f}m)")
            print(f"   LiDAR readings: {self.lidar_data}")
            print(f"   Robot position: [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}]")
            rospy.logerr(f"COLLISION! Min distance: {min_distance:.3f}m < {self.collision_threshold:.3f}m")
            return True
        return False
    
    def calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0
        # Reset goal reached flag for this step
        self.goal_reached_last_step = False
        
        # Distance reward
        current_distance = self.get_distance_to_goal()
        distance_reward = self.prev_distance_to_goal - current_distance
        reward += self.w_distance * distance_reward
        
        # Angle reward
        current_angle = self.get_angle_to_goal()
        angle_reward = self.prev_angle_to_goal - current_angle
        reward += self.w_angle * angle_reward
        
        # Velocity reward (encourage forward motion)
        velocity_reward = self.current_velocity[0] / self.max_velocity
        reward += self.w_velocity * velocity_reward
        
        # Strong penalty for backward movement
        if self.current_velocity[0] < -0.01:  # Moving backward
            backward_penalty = -10.0 * abs(self.current_velocity[0])
            reward += backward_penalty
            if self.episode_steps % 100 == 0:  # Log occasionally
                rospy.logwarn(f"Backward movement detected! Velocity: {self.current_velocity[0]:.3f}, Penalty: {backward_penalty:.3f}")
        
        # Collision detection and penalty
        collision_detected = self.check_collision()
        if collision_detected:
            reward += self.w_collision
            rospy.logerr("COLLISION DETECTED - Episode terminated!")
        
        # Goal reached reward
        if current_distance < self.distance_threshold and current_angle < self.angle_threshold:
            reward += self.w_goal_reached
            self.goal_reached_last_step = True
            rospy.loginfo("Goal reached!")
        
        # Time penalty
        reward += self.w_time_penalty

        # Heavy penalty for long episodes: ramp between 20-25s
        elapsed_sec = (rospy.Time.now() - self.episode_start_time).to_sec()
        if elapsed_sec > 20.0:
            # over_frac ranges 0..1 for 20s..25s, clamps at 1 beyond 25s
            over_frac = min(elapsed_sec - 20.0, 5.0) / 5.0
            # Apply an extra per-step penalty up to -1.0 per step after 25s
            reward += -1.0 * over_frac
        
        # Episode termination conditions
        done = False
        
        # Check collision first (highest priority)
        if collision_detected:
            done = True
            rospy.logerr("🚨 EPISODE TERMINATED DUE TO COLLISION 🚨")
            print("🚨 EPISODE TERMINATED DUE TO COLLISION 🚨")
        
        # Check if episode should end
        current_time = rospy.Time.now()
        if (current_time - self.episode_start_time) > self.max_episode_time:
            done = True
            rospy.logerr("⏰ EPISODE TIMEOUT ⏰")
            print("⏰ EPISODE TIMEOUT ⏰")
        
        if self.episode_steps >= self.max_episode_steps:
            done = True
            rospy.loginfo("⏰ EPISODE MAX STEPS REACHED ⏰")
            print("⏰ EPISODE MAX STEPS REACHED ⏰")
        
        if current_distance < self.distance_threshold and current_angle < self.angle_threshold:
            done = True
            rospy.loginfo("✅ EPISODE COMPLETED SUCCESSFULLY ✅")
            print("✅ EPISODE COMPLETED SUCCESSFULLY ✅")
        
        return reward, done
    
    def publish_reward(self, reward_info):
        """Publish reward and done flag"""
        reward, done = reward_info
        # Accumulate episode return before potential reset
        self.episode_return += reward
        
        msg = Float32MultiArray()
        msg.data = [reward, float(done)]
        self.reward_pub.publish(msg)

        # If goal was reached this step, report total episode reward so far
        if self.goal_reached_last_step:
            rospy.loginfo(f"Goal reached! Episode reward: {self.episode_return:.3f}")
            print(f"✅ Goal reached. Episode reward: {self.episode_return:.3f}")
        
        # Reset episode if done
        if done:
            self.reset_episode()
    
    def reset_episode(self):
        """Reset episode tracking"""
        self.episode_start_time = rospy.Time.now()
        self.episode_steps = 0
        self.prev_distance_to_goal = float('inf')
        self.prev_angle_to_goal = 0.0
        self.prev_velocity = np.array([0.0, 0.0])
        self.episode_return = 0.0
        self.goal_reached_last_step = False
        self.episode_count += 1  # Increment episode count for testing
        rospy.loginfo(f"Episode reset (Episode {self.episode_count})")

if __name__ == "__main__":
    try:
        reward_func = RewardFunction()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 