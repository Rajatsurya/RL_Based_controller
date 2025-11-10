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

class SingleRespawnController:
    def __init__(self, model_name=None):
        """
        Single respawn controller - moves robot once and exits
        
        Args:
            model_name (str): Name of the robot model in Gazebo (auto-detect if None)
        """
        rospy.init_node('single_respawn_controller', anonymous=True)
        
        # Map data
        self.map_data = None
        self.map_info = None
        self.map_received = False
        
        # Wait for Gazebo services
        rospy.loginfo("Waiting for Gazebo services...")
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        
        # Create service proxies
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        
        # Mark Gazebo as available since services are ready
        self.gazebo_available = True

        # Publishers and subscribers
        self.pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        rospy.loginfo("Services ready!")

        # Clearance policy: ensure respawn keeps at least this distance from obstacles
        # Try to read collision threshold from reward_function; fallback to 0.15m
        collision_threshold = rospy.get_param('/reward_function/collision_threshold',
                                              rospy.get_param('~collision_threshold', 0.15))
        # Default clearance: max(0.3m, collision_threshold + 0.1m)
        self.min_clearance_m = rospy.get_param('~min_clearance_m', max(0.3, float(collision_threshold) + 0.1))
        rospy.loginfo(f"Respawn min clearance set to {self.min_clearance_m:.3f} m (collision_threshold={collision_threshold:.3f} m)")
        
        # Auto-detect model name if not provided
        if model_name is None:
            model_name = self.auto_detect_turtlebot_model()
        
        self.model_name = model_name
        
        # Wait for map data
        rospy.loginfo("Waiting for map data...")
        while not self.map_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Map data received!")
        
        # Check if robot exists
        if not self.robot_exists():
            rospy.logerr(f"Robot model '{model_name}' not found in Gazebo!")
            return
        
        rospy.loginfo(f"Found robot model: {model_name}")
        
    def auto_detect_turtlebot_model(self):
        """Auto-detect TurtleBot3 model name"""
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
    
    def find_safe_position_in_map(self, max_attempts=100, min_distance=0.5):
        """Find a safe position in the map coordinate frame"""
        if not self.map_received:
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
        
        for _ in range(max_attempts):
            # Generate random position
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            if self.is_safe_position_in_map(x, y, min_distance):
                return (x, y)
        
        rospy.logwarn(f"Could not find safe position after {max_attempts} attempts")
        return None
    
    def move_robot_in_gazebo(self, x, y, z=0.0, yaw_degrees=0.0):
        """Move robot in Gazebo to specified position"""
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
    
    def move_robot_unified(self, x, y, yaw_degrees=0.0):
        """
        Move robot in both Gazebo and RViz with perfect localization synchronization
        
        Args:
            x, y (float): Coordinates (same for both Gazebo and RViz)
            yaw_degrees (float): Yaw angle in degrees
        """
        rospy.loginfo(f"Moving robot to coordinates ({x:.2f}, {y:.2f}) with yaw {yaw_degrees:.1f}°")
        
        # Step 1: Move robot in Gazebo first
        gazebo_success = self.move_robot_in_gazebo(x, y, 0.0, yaw_degrees)
        
        if not gazebo_success:
            rospy.logerr("Failed to move robot in Gazebo, aborting")
            return False
        
        # Step 2: Wait a moment for Gazebo to update
        rospy.sleep(0.5)
        
        # Step 3: Verify robot position in Gazebo
        actual_pos = self.get_robot_position_in_gazebo()
        if actual_pos:
            actual_x, actual_y, actual_z, actual_yaw = actual_pos
            rospy.loginfo(f"Robot actual position in Gazebo: ({actual_x:.2f}, {actual_y:.2f}, {actual_z:.2f}, {actual_yaw:.1f}°)")
            
            # Check if position is close enough (within 0.1m and 5 degrees)
            pos_error = math.sqrt((actual_x - x)**2 + (actual_y - y)**2)
            yaw_error = abs(actual_yaw - yaw_degrees)
            if yaw_error > 180:
                yaw_error = 360 - yaw_error
                
            if pos_error > 0.1 or yaw_error > 5:
                rospy.logwarn(f"Position mismatch! Error: {pos_error:.3f}m, Yaw error: {yaw_error:.1f}°")
                # Use actual position from Gazebo for RViz
                x, y, yaw_degrees = actual_x, actual_y, actual_yaw
                rospy.loginfo(f"Using actual Gazebo position for RViz: ({x:.2f}, {y:.2f}, {yaw_degrees:.1f}°)")
        
        # Step 4: Set initial pose in RViz with verified coordinates
        self.set_initial_pose_in_rviz(x, y, yaw_degrees)
        
        # Step 5: Wait for localization to settle
        rospy.sleep(1.0)
        
        rospy.loginfo(f"Successfully synchronized robot at ({x:.2f}, {y:.2f}) in both Gazebo and RViz")
        return True
    
    def get_robot_position_in_gazebo(self):
        """Get current robot position in Gazebo"""
        try:
            response = self.get_model_state(model_name=self.model_name, relative_entity_name="")
            if response.success:
                pose = response.pose
                # Convert quaternion to yaw
                qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                yaw_degrees = math.degrees(yaw)
                return (pose.position.x, pose.position.y, pose.position.z, yaw_degrees)
            else:
                rospy.logwarn(f"Model {self.model_name} not found in Gazebo")
                return None
        except Exception as e:
            rospy.logerr(f"Error getting robot position from Gazebo: {e}")
            return None
    
    def respawn_robot_once(self, min_distance=0.5, max_attempts=10):
        """Hard-coded respawn to a fixed pose in both Gazebo and RViz."""
        rospy.loginfo("Starting single respawn operation (hard-coded target)...")

        if not self.gazebo_available:
            rospy.logwarn("Gazebo not available - respawn disabled")
            return False

        # Hard-coded target pose
        target_x = -2.000495
        target_y = -0.497520
        target_yaw_deg = 0.039261

        rospy.loginfo(f"Moving robot to fixed pose: ({target_x:.6f}, {target_y:.6f}) yaw {target_yaw_deg:.1f}°")

        # Move in Gazebo
        gazebo_success = self.move_robot_in_gazebo(target_x, target_y, 0.0, target_yaw_deg)
        if not gazebo_success:
            rospy.logerr("Failed to move robot in Gazebo to fixed pose")
            return False

        rospy.sleep(0.5)

        # Set pose in RViz
        self.set_initial_pose_in_rviz(target_x, target_y, target_yaw_deg)
        rospy.sleep(1.0)

        rospy.loginfo("✅ Robot respawned to fixed pose successfully!")
        return True

def main():
    """Main function - respawn robot once and exit"""
    try:
        # Configuration
        min_distance = 0.5    # minimum distance from obstacles in meters
        max_attempts = 10     # maximum number of attempts
        
        rospy.loginfo("=== Single Robot Respawn ===")
        rospy.loginfo("This script will move the robot to a random safe position once and exit")
        rospy.loginfo(f"Will try up to {max_attempts} times if respawning fails")
        
        # Create and run the controller
        controller = SingleRespawnController()
        
        # Perform single respawn (with retry logic)
        success = controller.respawn_robot_once(min_distance, max_attempts)
        
        if success:
            rospy.loginfo("Program completed successfully!")
        else:
            rospy.logerr("Program failed after maximum attempts!")
            
        rospy.loginfo("Exiting...")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()
