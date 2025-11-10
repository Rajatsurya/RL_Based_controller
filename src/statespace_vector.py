#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionGoal
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Twist, PoseStamped
from std_msgs.msg import Float32MultiArray, Bool

class RLStateBuilder:
    def __init__(self):
        rospy.init_node("rl_state_builder")

        self.lidar_data = np.zeros(21)
        self.imu_data = np.zeros(2)   # pitch, yaw
        self.vel_data = np.zeros(2)   # linear.x, angular.z
        self.goal_rel = np.zeros(4)   # dx, dy, dyaw, distance

        # Internal robot state
        self.current_pos = np.zeros(2)
        self.current_yaw = 0.0
        self.odom_received = False  # Track if we've received valid odometry

        # Dynamic goal (updated via topic)
        self.goal_position = np.zeros(2)
        self.goal_yaw = 0.0
        self.goal_received = False  # Track if goal has been set

        # Read topics from params
        lidar_topic = rospy.get_param('~lidar_topic', '/scan')
        imu_topic = rospy.get_param('~imu_topic', '/imu')
        odom_topic = rospy.get_param('~odom_topic', '/odom')
        # Subscribe only to simple goal for consistency across components
        simple_goal_topic = rospy.get_param('~simple_goal_topic', '/move_base_simple/goal')

        # Subscribers
        rospy.Subscriber(lidar_topic, LaserScan, self.lidar_callback)
        rospy.Subscriber(imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber(simple_goal_topic, PoseStamped, self.simple_goal_callback)

        # Publishers
        self.state_pub = rospy.Publisher('/rl_state', Float32MultiArray, queue_size=1)

        # Episode control
        self.episode_active = False
        self.episode_step_index = 0
        rospy.Subscriber('/rl_episode_active', Bool, self.episode_active_callback)

        # Wait for localization before starting state publishing
        rospy.loginfo("Waiting 5 seconds for AMCL localization to stabilize...")
        rospy.sleep(5.0)
        rospy.loginfo("Starting state vector publishing")
        
        rospy.Timer(rospy.Duration(0.1), self.update_state)  # 10 Hz

    def episode_active_callback(self, msg: Bool):
        previously_active = self.episode_active
        self.episode_active = bool(msg.data)
        if self.episode_active and not previously_active:
            # New episode started
            self.episode_step_index = 0

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        ranges[np.isinf(ranges)] = msg.range_max
        step = len(ranges) // 21
        self.lidar_data = ranges[::step][:21]

    def imu_callback(self, msg: Imu):
        orientation_q = msg.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.imu_data = np.array([pitch, yaw])

    def odom_callback(self, msg: Odometry):
        # Position
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.current_pos = np.array([pos.x, pos.y])
        (_, _, self.current_yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        
        # Mark that we've received odometry data
        if not self.odom_received:
            self.odom_received = True
            rospy.loginfo(f"First odometry received: position=({pos.x:.3f}, {pos.y:.3f}), yaw={self.current_yaw:.3f}")

        # Velocity
        twist = msg.twist.twist
        self.vel_data = np.array([twist.linear.x, twist.angular.z])

        # Relative goal (based on latest received goal)
        if not self.goal_received:
            # No goal set yet - initialize goal to current robot position
            # This makes relative goal = [0, 0, 0, 0] until real goal arrives
            self.goal_position = self.current_pos.copy()
            self.goal_yaw = self.current_yaw
        
        dx = self.goal_position[0] - self.current_pos[0]
        dy = self.goal_position[1] - self.current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        dyaw = self.goal_yaw - self.current_yaw
        self.goal_rel = np.array([dx, dy, dyaw, distance])

    def simple_goal_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.goal_position = np.array([pos.x, pos.y])
        (_, _, self.goal_yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        
        if not self.goal_received:
            rospy.loginfo(f"First goal received: position=({pos.x:.3f}, {pos.y:.3f}), yaw={self.goal_yaw:.3f}")
        self.goal_received = True  # Mark that a goal has been received

    def update_state(self, event):
        state_vector = np.concatenate([
            self.lidar_data,
            self.imu_data,
            self.vel_data,
            self.goal_rel
        ])
        if state_vector.shape == (29,):
            # Publish state vector for TD3 agent
            state_msg = Float32MultiArray()
            state_msg.data = state_vector.tolist()
            self.state_pub.publish(state_msg)
            
            # Log only at steps 1, 500, and 1000 within an active episode (commented out for cleaner logs)
            if self.episode_active:
                self.episode_step_index += 1
                # if self.episode_step_index in (1, 500, 1000):
                #     rospy.loginfo(f"Step {self.episode_step_index}: State Vector (29x1): {state_vector.round(2)}")
        else:
            rospy.logwarn_throttle(5, f"State vector shape mismatch: {state_vector.shape}")

if __name__ == "__main__":
    try:
        RLStateBuilder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
