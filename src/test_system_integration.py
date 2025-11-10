#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseActionGoal

def test_system_integration():
    """Test script to verify robot movement control and goal generator"""
    rospy.init_node("test_system_integration")
    
    # Publishers
    td3_control_pub = rospy.Publisher('/td3_episode_control', Bool, queue_size=1)
    reward_pub = rospy.Publisher('/rl_reward', Float32MultiArray, queue_size=1)
    
    # Subscribers
    cmd_vel_received = False
    goal_received = False
    
    def cmd_vel_callback(msg):
        nonlocal cmd_vel_received
        cmd_vel_received = True
        rospy.loginfo(f"Received cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
    
    def goal_callback(msg):
        nonlocal goal_received
        goal_received = True
        pos = msg.goal.target_pose.pose.position
        rospy.loginfo(f"Received goal: x={pos.x:.3f}, y={pos.y:.3f}")
    
    # Subscribe to topics
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, goal_callback)
    
    rospy.loginfo("=== System Integration Test ===")
    rospy.loginfo("This script will test robot movement control and goal generation")
    rospy.loginfo("Make sure the training system is running!")
    
    # Wait for system to be ready
    rospy.sleep(3.0)
    
    # Test 1: Check if TD3 agent is responding to control commands
    rospy.loginfo("\n--- Test 1: TD3 Agent Control ---")
    rospy.loginfo("Testing TD3 agent start/stop control...")
    
    # Start episode
    rospy.loginfo("Sending start episode command...")
    td3_control_pub.publish(Bool(True))
    rospy.sleep(2.0)
    
    # Check if robot is moving
    if cmd_vel_received:
        rospy.loginfo("✓ TD3 Agent is responding - robot is moving")
    else:
        rospy.logwarn("✗ TD3 Agent not responding - robot not moving")
    
    # Stop episode
    rospy.loginfo("Sending stop episode command...")
    td3_control_pub.publish(Bool(False))
    rospy.sleep(2.0)
    
    # Test 2: Check goal generator
    rospy.loginfo("\n--- Test 2: Goal Generator ---")
    rospy.loginfo("Waiting for goals to be generated...")
    
    # Wait for goals
    start_time = time.time()
    while not goal_received and (time.time() - start_time) < 10.0:
        rospy.sleep(0.1)
    
    if goal_received:
        rospy.loginfo("✓ Goal generator is working - goals are being published")
    else:
        rospy.logwarn("✗ Goal generator not working - no goals received")
    
    # Test 3: Simulate collision and episode restart
    rospy.loginfo("\n--- Test 3: Collision Simulation ---")
    rospy.loginfo("Simulating collision episode...")
    
    # Start episode
    td3_control_pub.publish(Bool(True))
    rospy.sleep(1.0)
    
    # Simulate collision
    rospy.loginfo("Simulating collision...")
    reward_msg = Float32MultiArray()
    reward_msg.data = [-100.0, True]  # Collision reward, episode done
    reward_pub.publish(reward_msg)
    rospy.sleep(2.0)
    
    # Check if robot stopped
    cmd_vel_received = False
    rospy.sleep(1.0)
    if not cmd_vel_received:
        rospy.loginfo("✓ Robot stopped after collision simulation")
    else:
        rospy.logwarn("✗ Robot still moving after collision")
    
    # Test 4: Restart episode
    rospy.loginfo("Restarting episode...")
    td3_control_pub.publish(Bool(True))
    rospy.sleep(2.0)
    
    if cmd_vel_received:
        rospy.loginfo("✓ Robot restarted after episode restart")
    else:
        rospy.logwarn("✗ Robot not moving after restart")
    
    rospy.loginfo("\n=== Test Summary ===")
    rospy.loginfo("TD3 Agent Control: " + ("✓ Working" if cmd_vel_received else "✗ Not Working"))
    rospy.loginfo("Goal Generator: " + ("✓ Working" if goal_received else "✗ Not Working"))
    rospy.loginfo("Test completed!")

if __name__ == "__main__":
    try:
        test_system_integration()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
