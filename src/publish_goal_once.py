#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import sys

def publish_goal_once():
    """Publish a navigation goal once and exit"""
    rospy.init_node('goal_publisher_once', anonymous=True)

    # Get parameters
    goal_x = rospy.get_param('~goal_x', -0.4)
    goal_y = rospy.get_param('~goal_y', -0.4)
    goal_yaw = rospy.get_param('~goal_yaw', 0.039261)
    goal_frame = rospy.get_param('~goal_frame', 'map')
    delay = rospy.get_param('~delay', 2.0)  # Wait for other nodes to start
    repeats = rospy.get_param('~repeats', 5)

    # Create publisher
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)

    rospy.loginfo(f"Waiting {delay} seconds for nodes to initialize...")
    rospy.sleep(delay)

    # Create goal message
    goal = PoseStamped()
    goal.header.frame_id = goal_frame
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = goal_x
    goal.pose.position.y = goal_y
    goal.pose.position.z = 0.0

    # Convert yaw to quaternion
    quat = quaternion_from_euler(0.0, 0.0, goal_yaw)
    goal.pose.orientation.x = quat[0]
    goal.pose.orientation.y = quat[1]
    goal.pose.orientation.z = quat[2]
    goal.pose.orientation.w = quat[3]

    # Publish multiple times to ensure it's received
    rospy.loginfo(f"Publishing goal: x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f} rad")
    for i in range(repeats):
        goal.header.stamp = rospy.Time.now()
        goal_pub.publish(goal)
        rospy.sleep(0.1)

    rospy.loginfo("Goal published successfully!")

    # Keep node alive briefly to ensure message is sent
    rospy.sleep(0.5)

if __name__ == '__main__':
    try:
        publish_goal_once()
    except rospy.ROSInterruptException:
        pass
