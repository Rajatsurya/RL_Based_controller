#!/usr/bin/env python3

import rospy
import numpy as np
import random
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import OccupancyGrid, Odometry
import time

class GoalGenerator:
    def __init__(self):
        rospy.init_node("goal_generator")
        
        # Map boundaries will be calculated from actual map data
        self.map_center_x = 0.0  # Will be updated from map data
        self.map_center_y = 0.0
        self.map_bounds = None  # Will store actual map boundaries
        self.safety_margin = 0.15  # Safety margin from boundaries
        
        # Goal publishing parameters
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base/goal')
        self.goal_pub = rospy.Publisher(self.goal_topic, MoveBaseActionGoal, queue_size=1)
        
        # Goal generation parameters
        self.min_distance = rospy.get_param('~min_distance', 0.5)  # Minimum distance from current position
        self.max_goals = rospy.get_param('~max_goals', 1000)  # Maximum goals to generate
        self.goal_timeout = rospy.get_param('~goal_timeout', 30.0)  # Timeout for unreachable goals
        self.sample_retries = rospy.get_param('~sample_retries', 200)

        # Map-based sampling parameters
        self.use_map = rospy.get_param('~use_map', True)
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.free_max_value = rospy.get_param('~free_max_value', 0)  # cells <= this are considered free
        self.treat_unknown_as_free = rospy.get_param('~treat_unknown_as_free', False)
        self.keepout_radius_m = rospy.get_param('~keepout_radius_m', 0.15)  # margin away from obstacles
        self.use_odom_min_distance = rospy.get_param('~use_odom_min_distance', True)

        # Map state
        self.map_ready = False
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.occupancy = None  # numpy array shape (H, W)
        self.free_indices = None  # array of (y, x) indices for free cells

        # Robot state (for min_distance filtering)
        self.robot_x = None
        self.robot_y = None

        if self.use_map:
            rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_callback, queue_size=1)
            if self.use_odom_min_distance:
                rospy.Subscriber('/odom', Odometry, self._odom_callback, queue_size=5)
        
        # Current goal count
        self.goals_published = 0
        
        # Goal status tracking
        self.current_goal_active = False
        self.current_goal_start_time = None
        self.current_goal_x = None
        self.current_goal_y = None
        self.goal_reached_threshold = rospy.get_param('~goal_reached_threshold', 0.3)  # Distance threshold for goal reached
        self.goal_reached_delay = rospy.get_param('~goal_reached_delay', 3.0)  # Delay after goal reached before next goal
        
        # Start goal generation
        rospy.loginfo(f"Goal Generator initialized. Using map sampling: {self.use_map}")
        if not self.use_map:
            rospy.loginfo(f"Fallback radial sampling with safety margin: {self.safety_margin}")
        
        # Start timeout monitoring timer
        self.timeout_timer = rospy.Timer(rospy.Duration(1.0), self._check_goal_timeout)
        
        # Generate first goal immediately
        self.generate_goal()
    
    def generate_goal(self):
        """Generate and publish a random goal within safe boundaries"""
        if self.goals_published >= self.max_goals:
            rospy.loginfo("Maximum goals reached. Stopping goal generation.")
            self.timeout_timer.stop()
            return
        
        if self.current_goal_active:
            rospy.logdebug("Previous goal still active, waiting for completion or timeout")
            return
        
        # Generate random goal within map free space (or fallback)
        goal_x, goal_y = self.generate_random_position()
        goal_yaw = self.generate_random_yaw()
        
        # Create goal message
        goal_msg = MoveBaseActionGoal()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        
        # Set goal position
        goal_msg.goal.target_pose.header.stamp = rospy.Time.now()
        goal_msg.goal.target_pose.header.frame_id = "map"
        
        goal_msg.goal.target_pose.pose.position.x = goal_x
        goal_msg.goal.target_pose.pose.position.y = goal_y
        goal_msg.goal.target_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, goal_yaw)
        goal_msg.goal.target_pose.pose.orientation.x = quat[0]
        goal_msg.goal.target_pose.pose.orientation.y = quat[1]
        goal_msg.goal.target_pose.pose.orientation.z = quat[2]
        goal_msg.goal.target_pose.pose.orientation.w = quat[3]
        
        # Publish goal
        self.goal_pub.publish(goal_msg)
        self.goals_published += 1
        self.current_goal_active = True
        self.current_goal_start_time = time.time()
        self.current_goal_x = goal_x
        self.current_goal_y = goal_y
        
        rospy.loginfo(f"Published goal {self.goals_published}: ({goal_x:.2f}, {goal_y:.2f}, yaw: {goal_yaw:.2f})")
    
    def generate_random_position(self):
        """Generate a random position from map free space if available, else fallback to radial sampling."""
        if self.use_map and self.map_ready and self.free_indices is not None and len(self.free_indices) > 0:
            keepout_cells = 0
            if self.keepout_radius_m and self.map_resolution and self.map_resolution > 0:
                keepout_cells = int(np.ceil(self.keepout_radius_m / self.map_resolution))
            
            rospy.logdebug(f"Map sampling: {len(self.free_indices)} free cells, keepout_cells: {keepout_cells}")

            for attempt in range(self.sample_retries):
                iy, ix = self.free_indices[np.random.randint(0, len(self.free_indices))]
                if keepout_cells > 0 and not self._cell_is_keepout_free(ix, iy, keepout_cells):
                    continue

                wx, wy = self._map_to_world(ix, iy)

                if self.use_odom_min_distance and self.robot_x is not None and self.robot_y is not None:
                    if np.hypot(wx - self.robot_x, wy - self.robot_y) < self.min_distance:
                        continue

                rospy.logdebug(f"Successfully sampled goal from map at attempt {attempt + 1}")
                return wx, wy

            rospy.logwarn(f"Failed to sample a valid goal from map after {self.sample_retries} retries. Falling back to radial sampling.")
        else:
            if not self.use_map:
                rospy.logwarn("Map sampling disabled, using fallback method")
            elif not self.map_ready:
                rospy.logwarn("Map not ready, using fallback method")
            elif self.free_indices is None or len(self.free_indices) == 0:
                rospy.logwarn(f"No free cells found in map, using fallback method")

        # Fallback: random sampling within map boundaries
        if self.map_bounds is not None:
            for _ in range(self.sample_retries):
                x = random.uniform(self.map_bounds['min_x'], self.map_bounds['max_x'])
                y = random.uniform(self.map_bounds['min_y'], self.map_bounds['max_y'])
                if self.is_position_safe(x, y):
                    return x, y
            rospy.logwarn("Failed to find safe position within map boundaries after retries")
        
        # Last resort: return map center
        rospy.logwarn("Using map center as fallback goal")
        return self.map_center_x, self.map_center_y
    
    def generate_random_yaw(self):
        """Generate random yaw angle"""
        return random.uniform(-np.pi, np.pi)
    
    def is_position_safe(self, x, y):
        """Check if position is within safe boundaries"""
        # If map is available, use it for safety check
        if self.use_map and self.map_ready and self.occupancy is not None:
            ix, iy = self._world_to_map(x, y)
            if ix is None:
                return False
            if not self._cell_is_free(ix, iy):
                return False

            keepout_cells = 0
            if self.keepout_radius_m and self.map_resolution and self.map_resolution > 0:
                keepout_cells = int(np.ceil(self.keepout_radius_m / self.map_resolution))
            if keepout_cells > 0 and not self._cell_is_keepout_free(ix, iy, keepout_cells):
                return False

            return True

        # Fallback: check if within map boundaries
        if self.map_bounds is not None:
            return (self.map_bounds['min_x'] <= x <= self.map_bounds['max_x'] and 
                    self.map_bounds['min_y'] <= y <= self.map_bounds['max_y'])
        
        # Last resort: return True
        return True

    # ------------------------- Map helpers -------------------------
    def _map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int16).reshape((self.map_height, self.map_width))

        if self.treat_unknown_as_free:
            free_mask = (data <= self.free_max_value) | (data == -1)
        else:
            free_mask = (data <= self.free_max_value) & (data >= 0)

        self.occupancy = data
        self.free_indices = np.argwhere(free_mask)
        
        # Calculate actual map boundaries
        self._calculate_map_bounds()
        
        self.map_ready = True
        keepout_cells = 0
        if self.keepout_radius_m and self.map_resolution and self.map_resolution > 0:
            keepout_cells = int(np.ceil(self.keepout_radius_m / self.map_resolution))
        
        rospy.loginfo(f"Map loaded: {self.map_width}x{self.map_height}, resolution: {self.map_resolution:.3f}")
        rospy.loginfo(f"Map bounds: x=[{self.map_bounds['min_x']:.2f}, {self.map_bounds['max_x']:.2f}], y=[{self.map_bounds['min_y']:.2f}, {self.map_bounds['max_y']:.2f}]")
        rospy.loginfo(f"Free cells found: {len(self.free_indices)}")
        rospy.loginfo(f"Keepout radius: {self.keepout_radius_m}m ({keepout_cells} cells)")
        rospy.loginfo(f"Safety margin: {self.safety_margin}m")

    def _odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
    
    def _check_goal_reached(self):
        """Check if current goal has been reached based on distance"""
        if not self.current_goal_active or self.current_goal_x is None or self.current_goal_y is None:
            return
        
        if self.robot_x is None or self.robot_y is None:
            return
        
        # Calculate distance to current goal
        distance = np.sqrt((self.robot_x - self.current_goal_x)**2 + (self.robot_y - self.current_goal_y)**2)
        
        if distance <= self.goal_reached_threshold:
            rospy.loginfo(f"Goal {self.goals_published} reached! Distance: {distance:.2f}m")
            self.current_goal_active = False
            self.current_goal_start_time = None
            self.current_goal_x = None
            self.current_goal_y = None
            
            # Generate next goal after the specified delay
            rospy.loginfo(f"Waiting {self.goal_reached_delay} seconds before next goal...")
            rospy.sleep(self.goal_reached_delay)
            self.generate_goal()
    
    def _check_goal_timeout(self, event):
        """Check if current goal has timed out or been reached"""
        if not self.current_goal_active or self.current_goal_start_time is None:
            return
        
        # First check if goal has been reached
        self._check_goal_reached()
        
        # If goal is still active, check for timeout
        if self.current_goal_active:
            elapsed_time = time.time() - self.current_goal_start_time
            if elapsed_time >= self.goal_timeout:
                rospy.logwarn(f"Goal {self.goals_published} timed out after {elapsed_time:.1f} seconds")
                self.current_goal_active = False
                self.current_goal_start_time = None
                self.current_goal_x = None
                self.current_goal_y = None
                
                # Generate next goal after timeout
                rospy.sleep(0.5)  # Small delay before next goal
                self.generate_goal()
    
    def _calculate_map_bounds(self):
        """Calculate actual map boundaries from map data"""
        if self.map_resolution is None or self.map_width is None or self.map_height is None:
            return
        
        # Calculate world coordinates of map boundaries
        min_x = self.map_origin_x
        max_x = self.map_origin_x + self.map_width * self.map_resolution
        min_y = self.map_origin_y
        max_y = self.map_origin_y + self.map_height * self.map_resolution
        
        # Add safety margin
        self.map_bounds = {
            'min_x': min_x + self.safety_margin,
            'max_x': max_x - self.safety_margin,
            'min_y': min_y + self.safety_margin,
            'max_y': max_y - self.safety_margin
        }
        
        # Update map center
        self.map_center_x = (self.map_bounds['min_x'] + self.map_bounds['max_x']) / 2.0
        self.map_center_y = (self.map_bounds['min_y'] + self.map_bounds['max_y']) / 2.0

    def _world_to_map(self, wx, wy):
        if self.map_resolution is None:
            return None, None
        ix = int(np.floor((wx - self.map_origin_x) / self.map_resolution))
        iy = int(np.floor((wy - self.map_origin_y) / self.map_resolution))
        if ix < 0 or iy < 0 or ix >= self.map_width or iy >= self.map_height:
            return None, None
        return ix, iy

    def _map_to_world(self, ix, iy):
        wx = self.map_origin_x + (ix + 0.5) * self.map_resolution
        wy = self.map_origin_y + (iy + 0.5) * self.map_resolution
        return wx, wy

    def _cell_is_free(self, ix, iy):
        if self.occupancy is None:
            return False
        val = self.occupancy[iy, ix]
        if val == -1 and self.treat_unknown_as_free:
            return True
        return (val >= 0) and (val <= self.free_max_value)

    def _cell_is_keepout_free(self, ix, iy, keepout_cells):
        if keepout_cells <= 0:
            return self._cell_is_free(ix, iy)
        x0 = max(0, ix - keepout_cells)
        x1 = min(self.map_width - 1, ix + keepout_cells)
        y0 = max(0, iy - keepout_cells)
        y1 = min(self.map_height - 1, iy + keepout_cells)
        window = self.occupancy[y0:y1+1, x0:x1+1]
        if self.treat_unknown_as_free:
            occ_mask = (window > self.free_max_value)
        else:
            occ_mask = (window > self.free_max_value) | (window == -1)
        return not np.any(occ_mask)
    
    def set_goal_interval(self, interval):
        """Change goal generation interval"""
        self.goal_interval = interval
        self.timer.stop()
        self.timer = rospy.Timer(rospy.Duration(self.goal_interval), self.generate_goal)
        rospy.loginfo(f"Goal interval changed to {interval} seconds")
    
    def stop_generation(self):
        """Stop goal generation"""
        self.timer.stop()
        rospy.loginfo("Goal generation stopped")

if __name__ == "__main__":
    try:
        generator = GoalGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 