#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import SetBool, SetBoolResponse
import os
import pickle

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)#check if this is correct
        
        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        return q1

class TD3Agent:
    def __init__(self, state_dim=29, action_dim=2, max_action=1.0):
        rospy.init_node("td3_agent")
        
        # Read dimensions from ROS params (override defaults if provided)
        state_dim = rospy.get_param('~state_dim', state_dim)
        action_dim = rospy.get_param('~action_dim', action_dim)
        max_action = rospy.get_param('~max_action', max_action)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Hyperparameters (overridable via ROS params)
        self.discount = rospy.get_param('~discount', 0.99)
        self.tau = rospy.get_param('~tau', 0.005)
        self.policy_noise = rospy.get_param('~policy_noise', 0.2)
        self.noise_clip = rospy.get_param('~noise_clip', 0.5)
        self.policy_freq = rospy.get_param('~policy_freq', 2)
        # Warmup before training begins; also allows starting training earlier with smaller buffers
        self.warmup_steps = rospy.get_param('~warmup_steps', 5000)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = rospy.get_param('~batch_size', 256)
        
        # Training parameters
        self.total_it = 0
        self.epsilon = rospy.get_param('~epsilon', 1.0)
        self.epsilon_decay = rospy.get_param('~epsilon_decay', 0.995)
        self.epsilon_min = rospy.get_param('~epsilon_min', 0.01)
        
        # ROS publishers and subscribers
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/rl_state', Float32MultiArray, self.state_callback)
        self.reward_sub = rospy.Subscriber('/rl_reward', Float32MultiArray, self.reward_callback)
        self.episode_active_sub = rospy.Subscriber('/rl_episode_active', Bool, self.episode_active_callback)
        
        
        # Current state
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Episode control
        self.episode_active = True
        self.episode_done = False
        
        # Training mode
        self.training_mode = rospy.get_param('~training_mode', True)
        self.model_path = rospy.get_param('~model_path', '/tmp/td3_model.pth')
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
            rospy.loginfo("Loaded existing model")
        
        rospy.loginfo("TD3 Agent initialized")
    
    def state_callback(self, msg):
        """Callback for receiving state vector from statespace_vector.py"""
        self.current_state = np.array(msg.data)
        
        if self.training_mode and self.episode_active and not self.episode_done:
            # Get action from TD3
            action = self.select_action(self.current_state)
            
            # Publish action
            self.publish_action(action)
            
            # Store experience (will be completed when reward is received)
            if self.last_action is not None:
                experience = (self.last_state, self.last_action, self.last_reward, 
                           self.current_state, False)  # done=False for now
                self.replay_buffer.append(experience)
                
                # Train only after warmup and if enough samples for a batch
                if len(self.replay_buffer) >= max(self.batch_size, self.warmup_steps):
                    self.train()
            
            self.last_state = self.current_state.copy()
            self.last_action = action
            self.episode_steps += 1
            # Decay epsilon every step (independent of training), bounded by epsilon_min
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            # Episode not active or done - stop robot
            self.stop_robot()
    
    def reward_callback(self, msg: Float32MultiArray):
        """Callback for receiving reward and done flag"""
        if len(msg.data) >= 2:
            reward = float(msg.data[0])
            done = bool(msg.data[1])
            self.update_reward(reward, done)
            
            # Stop robot if episode is done
            if done:
                self.episode_done = True
                self.episode_active = False
                self.stop_robot()
                rospy.loginfo("Episode ended - robot stopped")

    def episode_active_callback(self, msg: Bool):
        """External control: trainer toggles episode active state."""
        self.episode_active = bool(msg.data)
        if self.episode_active:
            # Reset done flag and per-episode counters on new episode start
            self.episode_done = False
            self.episode_steps = 0
            self.episode_reward = 0.0
            rospy.loginfo("TD3Agent: Episode activated by trainer; resetting episode_done and counters.")
        else:
            self.stop_robot()
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)
        else:
            # Policy action
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        return action
    
    def publish_action(self, action):
        """Publish action as Twist message"""
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        
        # Log backward movement for monitoring (not preventing it)
        if action[0] < -0.01:
            rospy.logwarn_throttle(2.0, f"Backward movement: linear.x={action[0]:.3f}")
        
        self.cmd_vel_pub.publish(twist)
        # Throttled debug to verify agent is publishing commands (commented out for cleaner logs)
        # try:
        #     rospy.loginfo_throttle(0.5, f"cmd_vel({self.cmd_vel_topic}) published: linear.x={twist.linear.x:.3f}, angular.z={twist.angular.z:.3f}, episode_active={self.episode_active and not self.episode_done}")
        # except Exception:
        #     pass

    
    
    def stop_robot(self):
        """Stop robot by publishing zero velocity"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
    
    
    def train(self):
        """Train the TD3 agent"""
        self.total_it += 1
        
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            rospy.logwarn_throttle(10, f"Not enough samples in replay buffer ({len(self.replay_buffer)} < {self.batch_size}). Skipping training.")
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        action = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        reward = torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        done = torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_reward(self, reward, done=False):
        """Update the last reward and handle episode completion"""
        self.last_reward = reward
        self.episode_reward += reward
        
        if done:
            # Update the last experience with done=True
            if len(self.replay_buffer) > 0:
                last_exp = self.replay_buffer[-1]
                self.replay_buffer[-1] = (last_exp[0], last_exp[1], last_exp[2], last_exp[3], True)
            
            rospy.loginfo(f"Episode finished. Total reward: {self.episode_reward:.2f}, Steps: {self.episode_steps}")
            # Save model on episode completion
            try:
                self.save_model()
            except Exception as e:
                rospy.logwarn(f"Failed to save model: {e}")
            self.episode_reward = 0.0
            self.episode_steps = 0
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_path
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_it': self.total_it
        }, path)
        rospy.loginfo(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_it = checkpoint.get('total_it', 0)
        
        rospy.loginfo(f"Model loaded from {path}")

if __name__ == "__main__":
    try:
        agent = TD3Agent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 