#!/usr/bin/env python3

import rospy
from std_srvs.srv import SetBool

def test_td3_service():
    """Test script to check TD3 service communication"""
    rospy.init_node("test_td3_service")
    
    try:
        # Wait for service
        rospy.loginfo("Waiting for TD3 episode control service...")
        rospy.wait_for_service('/td3_episode_control', timeout=10.0)
        
        # Create service proxy
        service = rospy.ServiceProxy('/td3_episode_control', SetBool)
        
        # Test start episode
        rospy.loginfo("Testing start episode...")
        response = service(True)
        rospy.loginfo(f"Start episode response: success={response.success}, message='{response.message}'")
        
        rospy.sleep(2.0)
        
        # Test stop episode
        rospy.loginfo("Testing stop episode...")
        response = service(False)
        rospy.loginfo(f"Stop episode response: success={response.success}, message='{response.message}'")
        
        rospy.loginfo("Service test completed successfully!")
        
    except rospy.ROSException as e:
        rospy.logerr(f"Service test failed: {e}")
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == "__main__":
    test_td3_service()
