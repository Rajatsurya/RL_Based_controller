#!/usr/bin/env python3

import rospy
import time
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import OccupancyGrid

def test_goal_generator_with_map():
    """Test script to verify goal generator works with map data"""
    rospy.init_node("test_goal_generator_map")
    
    goal_count = 0
    goals = []
    map_received = False
    
    def goal_callback(msg):
        nonlocal goal_count
        goal_count += 1
        pos = msg.goal.target_pose.pose.position
        goals.append((pos.x, pos.y))
        rospy.loginfo(f"Goal {goal_count}: x={pos.x:.3f}, y={pos.y:.3f}")
    
    def map_callback(msg):
        nonlocal map_received
        map_received = True
        rospy.loginfo(f"Map received: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution:.3f}")
        rospy.loginfo(f"Map origin: ({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})")
    
    # Subscribe to goals and map
    rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, goal_callback)
    rospy.Subscriber('/map', OccupancyGrid, map_callback)
    
    rospy.loginfo("=== Goal Generator Map Test ===")
    rospy.loginfo("Waiting for map data...")
    
    # Wait for map data
    start_time = time.time()
    while not map_received and (time.time() - start_time) < 10.0:
        rospy.sleep(0.1)
    
    if not map_received:
        rospy.logwarn("✗ No map data received within 10 seconds")
        rospy.logwarn("Make sure map_server is running and publishing to /map topic")
        return
    
    rospy.loginfo("✓ Map data received, waiting for goals...")
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
            
            # Check if goals are within reasonable map bounds
            if all(-20 < x < 20 and -20 < y < 20 for x, y in goals):
                rospy.loginfo("✓ Goals are within reasonable map bounds")
            else:
                rospy.logwarn("⚠ Some goals may be outside map bounds")
                
            # Check goal distribution
            if len(set(goals)) > 1:
                rospy.loginfo("✓ Goals are being generated at different locations")
            else:
                rospy.logwarn("⚠ All goals are at the same location")
    else:
        rospy.logwarn("✗ Goal generator not working - no goals received")
        rospy.logwarn("Check if goal_generator.py is running and map data is available")

if __name__ == "__main__":
    try:
        test_goal_generator_with_map()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
