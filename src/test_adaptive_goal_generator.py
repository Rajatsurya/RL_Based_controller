#!/usr/bin/env python3

import rospy
import time
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import Twist

def test_adaptive_goal_generator():
    """Test script to debug the adaptive goal generator"""
    rospy.init_node("test_adaptive_goal_generator")
    
    goal_count = 0
    goals = []
    
    def goal_callback(msg):
        nonlocal goal_count
        goal_count += 1
        pos = msg.goal.target_pose.pose.position
        goals.append((pos.x, pos.y))
        rospy.loginfo(f"Goal {goal_count} received: x={pos.x:.3f}, y={pos.y:.3f}")
    
    def cmd_vel_callback(msg):
        linear_vel = abs(msg.linear.x)
        angular_vel = abs(msg.angular.z)
        if linear_vel > 0.01 or angular_vel > 0.01:
            rospy.loginfo(f"cmd_vel detected: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
    
    # Subscribe to goals and cmd_vel
    rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, goal_callback)
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    
    rospy.loginfo("=== Adaptive Goal Generator Test ===")
    rospy.loginfo("This test will monitor goals and cmd_vel for 30 seconds")
    rospy.loginfo("Make sure adaptive_goal_generator.py is running")
    
    # Wait for goals
    start_time = time.time()
    while (time.time() - start_time) < 30.0:
        rospy.sleep(0.1)
    
    rospy.loginfo(f"\n=== Test Results ===")
    rospy.loginfo(f"Total goals received: {goal_count}")
    
    if goal_count > 0:
        rospy.loginfo("✓ Adaptive goal generator is working!")
        if goals:
            x_coords = [g[0] for g in goals]
            y_coords = [g[1] for g in goals]
            rospy.loginfo(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
            rospy.loginfo(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
    else:
        rospy.logwarn("✗ No goals received - adaptive goal generator may not be working")
        rospy.logwarn("Check if adaptive_goal_generator.py is running and map data is available")

if __name__ == "__main__":
    try:
        test_adaptive_goal_generator()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
