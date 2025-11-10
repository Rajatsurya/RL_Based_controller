#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Float32MultiArray

def test_collision_simulation():
    """Test script to simulate collision detection and respawn integration"""
    rospy.init_node("collision_test")
    
    # Publisher to simulate reward messages
    reward_pub = rospy.Publisher('/rl_reward', Float32MultiArray, queue_size=1)
    
    rospy.loginfo("Collision Integration Test")
    rospy.loginfo("This script will simulate collision detection")
    rospy.loginfo("Make sure the training system is running!")
    
    # Wait for system to be ready
    rospy.sleep(2.0)
    
    # Simulate normal episode for a few steps
    rospy.loginfo("Simulating normal episode...")
    for i in range(5):
        msg = Float32MultiArray()
        msg.data = [1.0, False]  # Normal reward, episode not done
        reward_pub.publish(msg)
        rospy.sleep(1.0)
    
    # Simulate collision
    rospy.loginfo("Simulating COLLISION...")
    msg = Float32MultiArray()
    msg.data = [-100.0, True]  # Collision reward, episode done
    reward_pub.publish(msg)
    
    rospy.loginfo("Collision simulated! Check if respawn process starts...")
    rospy.loginfo("The system should:")
    rospy.loginfo("1. Detect collision (negative reward)")
    rospy.loginfo("2. Execute single_respawn.py")
    rospy.loginfo("3. Wait 5 seconds")
    rospy.loginfo("4. Start next episode")
    
    # Wait and observe
    rospy.sleep(10.0)
    
    # Simulate another normal episode
    rospy.loginfo("Simulating another normal episode...")
    for i in range(3):
        msg = Float32MultiArray()
        msg.data = [2.0, False]  # Normal reward, episode not done
        reward_pub.publish(msg)
        rospy.sleep(1.0)
    
    # Simulate normal episode end
    msg = Float32MultiArray()
    msg.data = [50.0, True]  # Good reward, episode done normally
    reward_pub.publish(msg)
    
    rospy.loginfo("Test completed!")

if __name__ == "__main__":
    try:
        test_collision_simulation()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
