#!/usr/bin/env python3

import rospy
import time
from move_base_msgs.msg import MoveBaseActionGoal

def test_goal_generator():
    """Test script to verify goal generator is working"""
    rospy.init_node("test_goal_generator")
    
    goal_count = 0
    goals = []
    
    def goal_callback(msg):
        nonlocal goal_count
        goal_count += 1
        pos = msg.goal.target_pose.pose.position
        goals.append((pos.x, pos.y))
        rospy.loginfo(f"Goal {goal_count}: x={pos.x:.3f}, y={pos.y:.3f}")
    
    # Subscribe to goals
    rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, goal_callback)
    
    rospy.loginfo("=== Goal Generator Test ===")
    rospy.loginfo("Waiting for goals to be generated...")
    rospy.loginfo("This test will run for 30 seconds")
    
    # Wait for goals
    start_time = time.time()
    while (time.time() - start_time) < 30.0:
        rospy.sleep(0.1)
    
    rospy.loginfo(f"\n=== Goal Generator Results ===")
    rospy.loginfo(f"Total goals received: {goal_count}")
    
    if goal_count > 0:
        rospy.loginfo("✓ Goal generator is working!")
        
        # Check if goals are within reasonable bounds
        if goals:
            x_coords = [g[0] for g in goals]
            y_coords = [g[1] for g in goals]
            rospy.loginfo(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
            rospy.loginfo(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
            
            # Check if goals are within reasonable map bounds (adjust as needed)
            if all(-10 < x < 10 and -10 < y < 10 for x, y in goals):
                rospy.loginfo("✓ Goals are within reasonable map bounds")
            else:
                rospy.logwarn("⚠ Some goals may be outside map bounds")
    else:
        rospy.logwarn("✗ Goal generator not working - no goals received")
        rospy.logwarn("Check if goal_generator.py is running and map data is available")

if __name__ == "__main__":
    try:
        test_goal_generator()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
