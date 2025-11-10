#!/usr/bin/env python3

import rospy
import numpy as np
import time
import threading
from std_msgs.msg import Float32MultiArray, Bool
import subprocess
import signal

class TrainingLauncher:
    def __init__(self):
        rospy.init_node('training_launcher')
        
        # Training parameters
        self.training_episodes = rospy.get_param('~training_episodes', 1000)
        # Progressive timeout strategy (no longer used as fixed value)
        self.episode_timeout = rospy.get_param('~episode_timeout', 60.0)  # Fallback default
        self.save_interval = rospy.get_param('~save_interval', 100)
        self.respawn_wait_time = rospy.get_param('~respawn_wait_time', 5.0)  # Wait time after respawn
        self.episode_interval_sec = rospy.get_param('~episode_interval_sec', 5.0)  # Wait time before next episode
        
        # Progressive timeout configuration
        # Episodes 1-200: 5 minutes, 201-600: 3 minutes, 601+: 1 minute
        self.timeout_phase_1_episodes = 200
        self.timeout_phase_2_episodes = 600
        self.timeout_phase_1_duration = 300.0  # 5 minutes
        self.timeout_phase_2_duration = 180.0  # 3 minutes
        self.timeout_phase_3_duration = 60.0   # 1 minute
        
        rospy.loginfo("Progressive Timeout Strategy Enabled:")
        rospy.loginfo(f"  Episodes 1-{self.timeout_phase_1_episodes}: {self.timeout_phase_1_duration/60:.1f} minutes")
        rospy.loginfo(f"  Episodes {self.timeout_phase_1_episodes+1}-{self.timeout_phase_2_episodes}: {self.timeout_phase_2_duration/60:.1f} minutes")
        rospy.loginfo(f"  Episodes {self.timeout_phase_2_episodes+1}+: {self.timeout_phase_3_duration/60:.1f} minutes")
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        
        # Episode control
        self.episode_active = False
        self.waiting_for_respawn = False
        self.collision_count = 0
        self.last_episode_was_collision = False
        
        # Subscribers
        rospy.Subscriber('/rl_reward', Float32MultiArray, self.reward_callback)
        # Publisher to inform agent of episode active state
        self.episode_active_pub = rospy.Publisher('/rl_episode_active', Bool, queue_size=1, latch=True)
        
        # Training statistics
        self.episode_start_time = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        rospy.loginfo("Training Launcher initialized")
        rospy.loginfo(f"Training for {self.training_episodes} episodes")
        
        # Start training loop
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        # Kick off the very first episode immediately
        self.schedule_next_episode(delay=0.0)
    
    def reward_callback(self, msg):
        """Callback for reward updates"""
        if len(msg.data) >= 2:
            reward = msg.data[0]
            done = bool(msg.data[1])
            
            if self.episode_active:
                self.episode_reward += reward
                self.episode_steps += 1
                
                # Check if this is a collision (very negative reward)
                if reward <= -50.0:  # Collision threshold
                    self.last_episode_was_collision = True
                    self.collision_count += 1
                    rospy.logwarn(f"🚨 COLLISION DETECTED! Episode {self.current_episode + 1} - Collision #{self.collision_count}")
                
                if done:
                    rospy.loginfo(f"Episode {self.current_episode + 1} ended")
                    self.end_episode()
    
    def end_episode(self):
        """Handle episode termination"""
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        # Log episode statistics
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        current_timeout = self.get_episode_timeout()
        
        rospy.loginfo("=" * 60)
        if self.last_episode_was_collision:
            rospy.logerr(f"💥 EPISODE {self.current_episode + 1} ENDED - COLLISION 💥")
        else:
            rospy.loginfo(f"✅ EPISODE {self.current_episode + 1} COMPLETED ✅")
        rospy.loginfo(f"  Reward: {self.episode_reward:.2f}")
        rospy.loginfo(f"  Steps: {self.episode_steps}")
        rospy.loginfo(f"  Time: {episode_time:.1f}s / {current_timeout:.0f}s timeout ({current_timeout/60:.1f}min)")
        rospy.loginfo(f"  Avg Reward (last 100): {avg_reward:.2f}")
        rospy.loginfo(f"  Best Reward: {self.best_reward:.2f}")
        rospy.loginfo(f"  Total Collisions: {self.collision_count}")
        rospy.loginfo("=" * 60)
        
        # Store episode data
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_steps)
        
        # Update best reward
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        
        # Reset episode tracking for next episode
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # End current episode
        self.episode_active = False
        try:
            self.episode_active_pub.publish(Bool(data=False))
        except Exception:
            pass
        
        # Advance trainer episode index
        self.current_episode += 1
        
        # Schedule next episode
        self.schedule_next_episode()
    
    def schedule_next_episode(self, delay=1.0):
        """Schedule the start of the next episode after delay"""
        def start_next_episode():
            time.sleep(delay)
            if not rospy.is_shutdown():
                # Always respawn before starting the next episode (after the first)
                if self.current_episode > 0:
                    self.waiting_for_respawn = True
                    rospy.loginfo("🔄 Respawning robot to initial pose before next episode...")
                    # Run single_respawn.py as a separate ROS node/process and wait (short timeout)
                    self.run_single_respawn_and_wait(timeout_sec=15.0)
                    self.waiting_for_respawn = False
                    # Clear collision flag after respawn
                    self.last_episode_was_collision = False
                
                # Always wait a fixed interval (default 5s) before starting the episode
                if self.episode_interval_sec and self.episode_interval_sec > 0:
                    rospy.loginfo(f"⏳ Waiting {self.episode_interval_sec} seconds before starting the episode...")
                    time.sleep(self.episode_interval_sec)

                # Mark episode active BEFORE goal generation, so the agent can move upon receiving /rl_state
                rospy.loginfo("===== Starting next episode (agent active) =====")
                self.episode_active = True
                self.episode_start_time = time.time()
                try:
                    self.episode_active_pub.publish(Bool(data=True))
                except Exception:
                    pass

                rospy.loginfo("Calling goal generator...")
                # Trigger goal generation process and wait for movement detection (process exit)
                self.run_goal_generator_and_wait()
        
        episode_thread = threading.Thread(target=start_next_episode)
        episode_thread.daemon = True
        episode_thread.start()
    
    def get_episode_timeout(self):
        """Get timeout duration based on current episode number"""
        episode_num = self.current_episode + 1  # +1 because we're starting a new episode
        
        if episode_num <= self.timeout_phase_1_episodes:
            return self.timeout_phase_1_duration
        elif episode_num <= self.timeout_phase_2_episodes:
            return self.timeout_phase_2_duration
        else:
            return self.timeout_phase_3_duration

    def run_goal_generator_and_wait(self, timeout_sec=None):
        """Start the goal generator node for this episode and wait until it exits.
        It will publish a goal and terminate when robot movement is detected.
        """
        if timeout_sec is None:
            # Reasonable upper bound; goal generator will typically quit as soon as agent moves
            timeout_sec = 60.0
        try:
            rospy.loginfo("🚀 Launching per-episode goal generator...")
            # Pipe output so we can mirror goal_generator logs into this node's console
            proc = subprocess.Popen(
                ["rosrun", "td3_rl_controller_high_buffer_size_respawn", "goal_generator.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            start_time = time.time()
            while proc.poll() is None and not rospy.is_shutdown():
                # Stream logs
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    if line:
                        rospy.loginfo(f"[goal_generator] {line.strip()}")
                if (time.time() - start_time) > timeout_sec:
                    rospy.logwarn("Goal generator did not exit within timeout; terminating it.")
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    break
                time.sleep(0.1)
            # Drain any remaining output
            if proc.stdout is not None:
                for line in proc.stdout.readlines():
                    if line:
                        rospy.loginfo(f"[goal_generator] {line.strip()}")
            rospy.loginfo("🎯 Goal generator finished for this episode.")
        except FileNotFoundError:
            rospy.logerr("rosrun not found. Ensure ROS environment is sourced.")
        except Exception as e:
            rospy.logerr(f"Failed to run goal generator: {e}")

    def run_single_respawn_and_wait(self, timeout_sec=15.0):
        """Invoke single_respawn.py to reposition the robot once, then return."""
        try:
            rospy.loginfo("🛠️ Running single_respawn.py for collision respawn...")
            proc = subprocess.Popen(
                ["rosrun", "td3_rl_controller_high_buffer_size_respawn", "single_respawn.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            start_time = time.time()
            while proc.poll() is None and not rospy.is_shutdown():
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    if line:
                        rospy.loginfo(f"[single_respawn] {line.strip()}")
                if (time.time() - start_time) > timeout_sec:
                    rospy.logwarn("single_respawn.py did not finish within timeout; terminating it.")
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    break
                time.sleep(0.1)
            if proc.stdout is not None:
                for line in proc.stdout.readlines():
                    if line:
                        rospy.loginfo(f"[single_respawn] {line.strip()}")
            rospy.loginfo("✅ single_respawn completed (or timed out); proceeding.")
        except FileNotFoundError:
            rospy.logerr("rosrun not found. Ensure ROS environment is sourced.")
        except Exception as e:
            rospy.logerr(f"Failed to run single_respawn.py: {e}")
    
    def training_loop(self):
        """Main training loop"""
        last_timeout_phase = None
        
        while not rospy.is_shutdown():
            # Wait for episode to be active
            if not self.episode_active:
                time.sleep(0.1)
                continue
            
            # Start new episode
            if self.episode_start_time is None:
                self.episode_start_time = time.time()
                self.episode_reward = 0.0
                self.episode_steps = 0
                
                # Get current episode timeout
                current_timeout = self.get_episode_timeout()
                
                # Log timeout phase changes
                current_phase = None
                if self.current_episode + 1 <= self.timeout_phase_1_episodes:
                    current_phase = 1
                elif self.current_episode + 1 <= self.timeout_phase_2_episodes:
                    current_phase = 2
                else:
                    current_phase = 3
                
                if current_phase != last_timeout_phase:
                    rospy.loginfo(f"🔄 Timeout Phase {current_phase}: {current_timeout/60:.1f} minutes per episode")
                    last_timeout_phase = current_phase
                
                rospy.loginfo(f"Starting episode {self.current_episode + 1} (timeout: {current_timeout/60:.1f} min)")
            
            # Wait for episode to complete or timeout
            current_timeout = self.get_episode_timeout()
            start_time = time.time()
            while (time.time() - start_time) < current_timeout:
                if rospy.is_shutdown():
                    return
                if not self.episode_active:
                    break  # Episode ended
                time.sleep(0.1)
            
            # Episode timeout
            if self.episode_active:
                rospy.logwarn(f"Episode {self.current_episode + 1} timed out after {current_timeout/60:.1f} minutes")
                self.end_episode()
    
    def save_training_progress(self):
        """Save training progress and model"""
        try:
            # Save training statistics
            stats = {
                'episode': self.current_episode,
                'rewards': self.episode_rewards,
                'lengths': self.episode_lengths,
                'best_reward': self.best_reward
            }
            
            # Save to file
            import json
            with open('/tmp/training_progress.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            rospy.loginfo(f"Training progress saved at episode {self.current_episode}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save training progress: {e}")
    
    def save_final_model(self):
        """Save final model and statistics"""
        try:
            # Save final statistics
            final_stats = {
                'total_episodes': self.current_episode,
                'final_rewards': self.episode_rewards,
                'final_lengths': self.episode_lengths,
                'best_reward': self.best_reward,
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            }
            
            import json
            with open('/tmp/final_training_stats.json', 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            rospy.loginfo("Final training statistics saved")
            rospy.loginfo(f"Best reward achieved: {self.best_reward:.2f}")
            rospy.loginfo(f"Average reward: {final_stats['avg_reward']:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save final model: {e}")

if __name__ == '__main__':
    try:
        launcher = TrainingLauncher()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Training launcher interrupted")
    except Exception as e:
        rospy.logerr(f"Training launcher error: {e}")
