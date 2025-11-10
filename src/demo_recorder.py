#!/usr/bin/env python3

import rospy
import rospkg
import os
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseGoal
from actionlib_msgs.msg import GoalID
from tf.transformations import quaternion_from_euler


class DemoRecorder:
    def __init__(self):
        rospy.init_node('demo_recorder')

        # Topics
        self.state_topic = rospy.get_param('~state_topic', '/rl_state')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')

        # Output directory
        default_dir = None
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path('td3_rl_controller_high_buffer_size_respawn')
            default_dir = os.path.join(pkg_path, 'demos')
        except Exception:
            default_dir = '/tmp/td3_demos'
        self.save_dir = rospy.get_param('~save_dir', default_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # Goal publishing configuration
        self.publish_goal = bool(rospy.get_param('~publish_goal', False))
        # Publish to both move_base action goal and simple goal if enabled
        self.publish_goal_to_move_base = bool(rospy.get_param('~publish_goal_to_move_base', True))
        self.publish_goal_to_simple = bool(rospy.get_param('~publish_goal_to_simple', True))
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base/goal')
        self.goal_simple_topic = rospy.get_param('~goal_simple_topic', '/move_base_simple/goal')
        self.goal_frame = rospy.get_param('~goal_frame', 'map')
        self.goal_x = float(rospy.get_param('~goal_x', 0.0))
        self.goal_y = float(rospy.get_param('~goal_y', 0.0))
        self.goal_yaw = float(rospy.get_param('~goal_yaw', 0.0))
        self.goal_repeats = int(rospy.get_param('~goal_repeats', 3))
        self.goal_pub = rospy.Publisher(self.goal_topic, MoveBaseActionGoal, queue_size=1, latch=True)
        self.goal_simple_pub = rospy.Publisher(self.goal_simple_topic, PoseStamped, queue_size=1, latch=True)

        # Initial pose publishing configuration (for AMCL)
        self.publish_initialpose = bool(rospy.get_param('~publish_initialpose', False))
        self.initialpose_frame = rospy.get_param('~initialpose_frame', 'map')
        self.initialpose_x = float(rospy.get_param('~initialpose_x', -2.000030))
        self.initialpose_y = float(rospy.get_param('~initialpose_y', -0.499938))
        self.initialpose_z = float(rospy.get_param('~initialpose_z', 0.0))
        self.initialpose_roll = float(rospy.get_param('~initialpose_roll', -0.000014))
        self.initialpose_pitch = float(rospy.get_param('~initialpose_pitch', 0.007706))
        self.initialpose_yaw = float(rospy.get_param('~initialpose_yaw', 0.000966))
        self.initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)

        # Recording control
        self.min_samples = int(rospy.get_param('~min_samples', 100))
        self.max_samples = int(rospy.get_param('~max_samples', 100000))

        # Buffers
        self.states = []
        self.actions = []
        self.timestamps = []

        self.latest_state = None

        rospy.Subscriber(self.state_topic, Float32MultiArray, self.state_cb, queue_size=10)
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_cb, queue_size=50)

        # Optionally publish initial pose and goal before recording
        if self.publish_initialpose:
            self.publish_initial_pose()
            rospy.sleep(0.2)
        if self.publish_goal:
            self.publish_fixed_goal()
            # Give state builder a moment to ingest the goal
            rospy.sleep(0.5)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(f"DemoRecorder started. Saving to: {self.save_dir}")
        rospy.spin()

    def state_cb(self, msg: Float32MultiArray):
        self.latest_state = np.array(msg.data, dtype=np.float32)

    def cmd_cb(self, msg: Twist):
        if self.latest_state is None:
            return
        if len(self.states) >= self.max_samples:
            return
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        self.states.append(self.latest_state.copy())
        self.actions.append(np.array([v, w], dtype=np.float32))
        self.timestamps.append(time.time())
        if len(self.states) % 500 == 0:
            rospy.loginfo(f"Recorded {len(self.states)} samples...")

    def on_shutdown(self):
        try:
            if len(self.states) < self.min_samples:
                rospy.logwarn(f"Not enough samples to save (have {len(self.states)}, need {self.min_samples})")
                return
            basename = time.strftime('demo_%Y%m%d_%H%M%S.npz')
            out_path = os.path.join(self.save_dir, basename)
            np.savez_compressed(
                out_path,
                states=np.stack(self.states, axis=0),
                actions=np.stack(self.actions, axis=0),
                timestamps=np.array(self.timestamps, dtype=np.float64),
            )
            rospy.loginfo(f"Saved demonstration: {out_path} (N={len(self.states)})")
        except Exception as e:
            rospy.logerr(f"Failed to save demonstration: {e}")

    def publish_fixed_goal(self):
        quat = quaternion_from_euler(0.0, 0.0, self.goal_yaw)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.goal_frame
        pose_stamped.pose.position.x = self.goal_x
        pose_stamped.pose.position.y = self.goal_y
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]

        mb_goal = MoveBaseGoal()
        mb_goal.target_pose = pose_stamped

        msg = MoveBaseActionGoal()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.goal_frame
        msg.goal_id = GoalID(stamp=rospy.Time.now(), id=f"demo_recorder_{int(time.time())}")
        msg.goal = mb_goal

        for i in range(max(1, self.goal_repeats)):
            if self.publish_goal_to_move_base:
                self.goal_pub.publish(msg)
            if self.publish_goal_to_simple:
                self.goal_simple_pub.publish(pose_stamped)
            rospy.sleep(0.05)
        rospy.loginfo(
            f"Published fixed goal (x={self.goal_x:.3f}, y={self.goal_y:.3f}, yaw={self.goal_yaw:.3f} rad) "
            f"to {'/move_base/goal' if self.publish_goal_to_move_base else ''}"
            f"{' and ' if self.publish_goal_to_move_base and self.publish_goal_to_simple else ''}"
            f"{'/move_base_simple/goal' if self.publish_goal_to_simple else ''}"
        )

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.initialpose_frame
        msg.pose.pose.position.x = self.initialpose_x
        msg.pose.pose.position.y = self.initialpose_y
        msg.pose.pose.position.z = self.initialpose_z
        q = quaternion_from_euler(self.initialpose_roll, self.initialpose_pitch, self.initialpose_yaw)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        # Reasonable covariance defaults (x, y, yaw)
        msg.pose.covariance = [0.25, 0, 0, 0, 0, 0,
                               0, 0.25, 0, 0, 0, 0,
                               0, 0, 0.0, 0, 0, 0,
                               0, 0, 0, 0.0, 0, 0,
                               0, 0, 0, 0, 0.0, 0,
                               0, 0, 0, 0, 0, 0.0685]
        self.initialpose_pub.publish(msg)
        rospy.loginfo(
            f"Published /initialpose: x={self.initialpose_x:.6f}, y={self.initialpose_y:.6f}, z={self.initialpose_z:.6f}, "
            f"rpy=({self.initialpose_roll:.6f}, {self.initialpose_pitch:.6f}, {self.initialpose_yaw:.6f})"
        )


if __name__ == '__main__':
    try:
        DemoRecorder()
    except rospy.ROSInterruptException:
        pass


