#!/usr/bin/env python3

import rospy
import math
import random
import numpy as np
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.srv import GetModelState, SetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState
from tf.transformations import quaternion_from_euler

class RespawnIntegration:
    def __init__(self, model_name=None):
        """
        Respawn integration for TD3 training - handles robot respawn on collision
        
        Args:
            model_name (str): Name of the robot model in Gazebo (auto-detect if None)
        """
        # Initialize a ROS node only if not already initialized (safe for in-process use)
        try:
            rospy.get_name()
        except rospy.ROSException:
            try:
                rospy.init_node('respawn_integration', anonymous=True)
            except rospy.exceptions.ROSException:
                pass
        
        # Map data
        self.map_data = None
        self.map_info = None
        self.map_received = False
        
        # Wait for Gazebo services
        rospy.loginfo("Waiting for Gazebo services...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/get_world_properties', timeout=5.0)
        except rospy.ROSException:
            rospy.logwarn("Gazebo services not available - respawn integration disabled")
            self.gazebo_available = False
            return
        
        # Create service proxies
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        
        # Publishers and subscribers
        self.pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        self.gazebo_available = True
        rospy.loginfo("Gazebo services ready!")
        
        # Auto-detect model name if not provided
        if model_name is None:
            model_name = self.auto_detect_turtlebot_model()
        
        self.model_name = model_name
        
        # Wait for map data
        rospy.loginfo("Waiting for map data...")
        timeout = rospy.Duration(10.0)  # 10 second timeout
        start_time = rospy.Time.now()
        while not self.map_received and not rospy.is_shutdown() and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if not self.map_received:
            rospy.logwarn("Map data not received within timeout - respawn may not work properly")
        else:
            rospy.loginfo("Map data received!")
        
        # Check if robot exists
        if self.gazebo_available and not self.robot_exists():
            rospy.logerr(f"Robot model '{model_name}' not found in Gazebo!")
            self.gazebo_available = False
        else:
            rospy.loginfo(f"Found robot model: {model_name}")
    
    def auto_detect_turtlebot_model(self):
        """Auto-detect TurtleBot3 model name"""
        if not self.gazebo_available:
            return "turtlebot3_burger"
            
        try:
            response = self.get_world_props()
            models = response.model_names
            
            # Look for turtlebot3 models
            turtlebot_models = [m for m in models if 'turtlebot3' in m.lower()]
            
            if turtlebot_models:
                model_name = turtlebot_models[0]
                rospy.loginfo(f"Auto-detected TurtleBot3 model: {model_name}")
                return model_name
            else:
                rospy.logwarn("No TurtleBot3 models found, using default: turtlebot3_burger")
                return "turtlebot3_burger"
                
        except Exception as e:
            rospy.logwarn(f"Error auto-detecting model: {e}, using default: turtlebot3_burger")
            return "turtlebot3_burger"
    
    def robot_exists(self):
        """Check if robot model exists in Gazebo"""
        if not self.gazebo_available:
            return False
            
        try:
            response = self.get_model_state(model_name=self.model_name, relative_entity_name="")
            return response.success
        except:
            return False
    
    def map_callback(self, msg):
        """Callback for map data"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.map_received = True
        rospy.loginfo("Map data received!")
    
    def is_safe_position_in_map(self, x, y, min_distance=0.5):
        """Check if a position is safe in the map coordinate frame"""
        if not self.map_received:
            return False
            
        # Convert world coordinates to map coordinates
        map_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        map_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Check bounds
        if (map_x < 0 or map_x >= self.map_info.width or 
            map_y < 0 or map_y >= self.map_info.height):
            return False
        
        # Calculate search radius in grid cells
        search_radius = int(min_distance / self.map_info.resolution)
        
        # Check area around the position
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                # Check bounds
                if (check_x < 0 or check_x >= self.map_info.width or 
                    check_y < 0 or check_y >= self.map_info.height):
                    continue
                
                # Check if cell is occupied or unknown
                cell_value = self.map_data[check_y, check_x]
                if cell_value > 50 or cell_value == -1:  # Occupied or unknown
                    # Check if this obstacle is within min_distance
                    distance = math.sqrt(dx*dx + dy*dy) * self.map_info.resolution
                    if distance < min_distance:
                        return False
        
        return True
    
    def find_safe_position_in_map(self, max_attempts=50, min_distance=0.5):
        """Find a safe position in the map coordinate frame"""
        if not self.map_received:
            rospy.logwarn("No map data available for respawn")
            return None
            
        # Get map bounds in world coordinates
        min_x = self.map_info.origin.position.x
        max_x = min_x + self.map_info.width * self.map_info.resolution
        min_y = self.map_info.origin.position.y
        max_y = min_y + self.map_info.height * self.map_info.resolution
        
        # Add some margin from edges
        margin = 1.0  # meters
        min_x += margin
        max_x -= margin
        min_y += margin
        max_y -= margin
        
        for attempt in range(max_attempts):
            # Generate random position
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            if self.is_safe_position_in_map(x, y, min_distance):
                return (x, y)
        
        rospy.logwarn(f"Could not find safe position after {max_attempts} attempts")
        return None
    
    def move_robot_in_gazebo(self, x, y, z=0.0, yaw_degrees=0.0):
        """Move robot in Gazebo to specified position"""
        if not self.gazebo_available:
            rospy.logwarn("Gazebo not available - cannot move robot")
            return False
            
        # Convert yaw to quaternion
        yaw_rad = math.radians(yaw_degrees)
        quat = quaternion_from_euler(0, 0, yaw_rad)
        
        # Create model state
        model_state = ModelState()
        model_state.model_name = self.model_name
        model_state.pose.position.x = x
        model_state.pose.position.y = y
        model_state.pose.position.z = z
        model_state.pose.orientation.x = quat[0]
        model_state.pose.orientation.y = quat[1]
        model_state.pose.orientation.z = quat[2]
        model_state.pose.orientation.w = quat[3]
        model_state.reference_frame = "world"
        
        try:
            response = self.set_model_state(model_state)
            if response.success:
                rospy.loginfo(f"Moved robot in Gazebo to ({x:.2f}, {y:.2f}, {z:.2f}) with yaw {yaw_degrees:.1f}°")
                return True
            else:
                rospy.logwarn(f"Failed to move robot in Gazebo: {response.status_message}")
                return False
        except Exception as e:
            rospy.logerr(f"Error moving robot in Gazebo: {e}")
            return False
    
    def set_initial_pose_in_rviz(self, x, y, yaw_degrees=0.0):
        """Set initial pose in RViz for navigation"""
        # Convert yaw to quaternion
        yaw_rad = math.radians(yaw_degrees)
        quat = quaternion_from_euler(0, 0, yaw_rad)
        
        # Create message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        # Set position
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = 0.0
        
        # Set orientation
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        
        # Set covariance
        pose_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  # x
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,  # y
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # z
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # roll
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # pitch
            0.0, 0.0, 0.0, 0.0, 0.0, 0.25   # yaw
        ]
        
        # Publish message
        self.pose_pub.publish(pose_msg)
        rospy.loginfo(f"Set initial pose in RViz: x={x:.2f}m, y={y:.2f}m, yaw={yaw_degrees:.1f}°")
    
    def respawn_robot(self, min_distance=0.5, max_attempts=10):
        """
        Respawn robot to a random safe position
        
        Args:
            min_distance (float): Minimum distance from obstacles in meters
            max_attempts (int): Maximum number of attempts before giving up
            
        Returns:
            bool: True if respawn successful, False otherwise
        """
        rospy.loginfo("🔄 Starting robot respawn...")
        
        if not self.gazebo_available:
            rospy.logwarn("Gazebo not available - respawn disabled")
            return False
        
        if not self.map_received:
            rospy.logwarn("Map data not available - respawn disabled")
            return False
        
        for attempt in range(max_attempts):
            rospy.loginfo(f"Respawn attempt {attempt + 1}/{max_attempts}")
            
            # Find safe position in map
            safe_pos = self.find_safe_position_in_map(min_distance=min_distance)
            
            if safe_pos is None:
                rospy.logwarn(f"Attempt {attempt + 1}: No safe position found, trying again...")
                rospy.sleep(0.5)  # Wait before retrying
                continue
            
            map_x, map_y = safe_pos
            yaw = random.uniform(0, 360)
            
            rospy.loginfo(f"Attempt {attempt + 1}: Found safe position: ({map_x:.2f}, {map_y:.2f}) with yaw {yaw:.1f}°")
            
            # Move robot in Gazebo
            gazebo_success = self.move_robot_in_gazebo(map_x, map_y, 0.0, yaw)
            
            if not gazebo_success:
                rospy.logwarn(f"Attempt {attempt + 1}: Failed to move robot in Gazebo, trying again...")
                rospy.sleep(0.5)
                continue
            
            # Wait a moment for Gazebo to update
            rospy.sleep(0.5)
            
            # Set initial pose in RViz
            self.set_initial_pose_in_rviz(map_x, map_y, yaw)
            
            # Wait for localization to settle
            rospy.sleep(1.0)
            
            rospy.loginfo("✅ Robot respawned successfully!")
            return True
        
        rospy.logerr(f"❌ Failed to respawn robot after {max_attempts} attempts!")
        return False

# Global instance for easy access
_respawn_integration = None

def get_respawn_integration():
    """Get the global respawn integration instance"""
    global _respawn_integration
    if _respawn_integration is None:
        _respawn_integration = RespawnIntegration()
    return _respawn_integration

def respawn_robot_on_collision(min_distance=0.5, max_attempts=10):
    """
    Convenience function to respawn robot on collision
    
    Args:
        min_distance (float): Minimum distance from obstacles in meters
        max_attempts (int): Maximum number of attempts before giving up
        
    Returns:
        bool: True if respawn successful, False otherwise
    """
    respawn = get_respawn_integration()
    return respawn.respawn_robot(min_distance, max_attempts)

if __name__ == '__main__':
    # Test the respawn integration
    try:
        respawn = RespawnIntegration()
        success = respawn.respawn_robot()
        if success:
            rospy.loginfo("Respawn integration test successful!")
        else:
            rospy.logerr("Respawn integration test failed!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
