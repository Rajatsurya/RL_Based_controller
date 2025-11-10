#!/usr/bin/env python3

import rospy
import time
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

def simple_goal_test():
    """Simple test to see if we can publish goals"""
    print("🚀 Starting simple goal test...")
    
    try:
        rospy.init_node("simple_goal_test")
        print("✅ ROS node initialized")
        
        # Create publisher
        goal_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=1)
        print("✅ Publisher created")
        
        # Wait a moment for publisher to register
        time.sleep(1.0)
        print("✅ Waiting for publisher to register...")
        
        # Create a simple goal
        goal_msg = MoveBaseActionGoal()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        
        # Set goal position
        goal_msg.goal.target_pose.header.stamp = rospy.Time.now()
        goal_msg.goal.target_pose.header.frame_id = "map"
        
        goal_msg.goal.target_pose.pose.position.x = 1.0
        goal_msg.goal.target_pose.pose.position.y = 1.0
        goal_msg.goal.target_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, 0)
        goal_msg.goal.target_pose.pose.orientation.x = quat[0]
        goal_msg.goal.target_pose.pose.orientation.y = quat[1]
        goal_msg.goal.target_pose.pose.orientation.z = quat[2]
        goal_msg.goal.target_pose.pose.orientation.w = quat[3]
        
        print("✅ Goal message created")
        
        # Publish goal
        goal_pub.publish(goal_msg)
        print("✅ Goal published: (1.0, 1.0, yaw: 0.0)")
        
        # Wait a bit
        time.sleep(2.0)
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_goal_test()
