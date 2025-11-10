#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Float32MultiArray

def test_collision_detection():
    """Test script to verify collision detection works"""
    rospy.init_node("collision_detection_test")
    
    # Publisher to simulate reward messages
    reward_pub = rospy.Publisher('/rl_reward', Float32MultiArray, queue_size=1)
    
    rospy.loginfo("=== Collision Detection Test ===")
    rospy.loginfo("This script will test collision detection and respawn system")
    rospy.loginfo("Make sure the training system is running!")
    
    # Wait for system to be ready
    rospy.sleep(3.0)
    
    # Test 1: Normal episode
    rospy.loginfo("\n--- Test 1: Normal Episode ---")
    rospy.loginfo("Simulating normal episode (no collision)...")
    for i in range(10):
        msg = Float32MultiArray()
        msg.data = [1.0, False]  # Normal reward, episode not done
        reward_pub.publish(msg)
        rospy.sleep(0.5)
    
    # End normal episode
    msg = Float32MultiArray()
    msg.data = [50.0, True]  # Good reward, episode done normally
    reward_pub.publish(msg)
    rospy.loginfo("Normal episode ended")
    
    rospy.sleep(2.0)
    
    # Test 2: Collision episode
    rospy.loginfo("\n--- Test 2: Collision Episode ---")
    rospy.loginfo("Simulating collision episode...")
    for i in range(5):
        msg = Float32MultiArray()
        msg.data = [1.0, False]  # Normal reward, episode not done
        reward_pub.publish(msg)
        rospy.sleep(0.5)
    
    # Simulate collision
    msg = Float32MultiArray()
    msg.data = [-100.0, True]  # Collision reward, episode done
    reward_pub.publish(msg)
    rospy.loginfo("COLLISION DETECTED! Episode terminated due to collision.")
    
    rospy.loginfo("\nExpected behavior:")
    rospy.loginfo("1. Collision handler should detect collision")
    rospy.loginfo("2. single_respawn.py should be executed")
    rospy.loginfo("3. Robot should be moved to safe location")
    rospy.loginfo("4. System should wait 5 seconds")
    rospy.loginfo("5. Next episode should start")
    
    # Wait and observe
    rospy.loginfo("\nObserving system behavior for 15 seconds...")
    rospy.sleep(15.0)
    
    # Test 3: Another normal episode
    rospy.loginfo("\n--- Test 3: Another Normal Episode ---")
    rospy.loginfo("Simulating another normal episode...")
    for i in range(5):
        msg = Float32MultiArray()
        msg.data = [2.0, False]  # Normal reward, episode not done
        reward_pub.publish(msg)
        rospy.sleep(0.5)
    
    # End normal episode
    msg = Float32MultiArray()
    msg.data = [75.0, True]  # Good reward, episode done normally
    reward_pub.publish(msg)
    rospy.loginfo("Normal episode ended")
    
    rospy.loginfo("\n=== Test Completed ===")
    rospy.loginfo("Check the logs to verify collision detection and respawn worked correctly")

if __name__ == "__main__":
    try:
        test_collision_detection()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
