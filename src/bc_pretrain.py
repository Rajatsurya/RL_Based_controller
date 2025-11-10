#!/usr/bin/env python3

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import rospy
from td3_agent import Actor, Critic


def load_demos(demos_dir):
    files = sorted(glob.glob(os.path.join(demos_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f"No .npz demos found in {demos_dir}")
    states_list, actions_list = [], []
    for f in files:
        data = np.load(f)
        states = np.asarray(data['states'], dtype=np.float32)
        actions = np.asarray(data['actions'], dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(f"Bad shapes in {f}: states {states.shape}, actions {actions.shape}")
        states_list.append(states)
        actions_list.append(actions)
    X = np.concatenate(states_list, axis=0)
    Y = np.concatenate(actions_list, axis=0)
    return X, Y


class BehaviorCloningTrainer:
    def __init__(self):
        rospy.init_node('bc_pretrain')
        self.demos_dir = rospy.get_param('~demos_dir', '/tmp/td3_demos')
        self.model_path = rospy.get_param('~model_path', '/tmp/td3_model.pth')
        self.epochs = int(rospy.get_param('~epochs', 10))
        self.batch_size = int(rospy.get_param('~batch_size', 256))#this is too large 
        self.lr = float(rospy.get_param('~lr', 3e-4))
        self.max_action = float(rospy.get_param('~max_action', 0.22))

        X, Y = load_demos(self.demos_dir)
        state_dim = int(rospy.get_param('~state_dim', X.shape[1]))
        action_dim = int(rospy.get_param('~action_dim', 2))

        # Models (CPU)
        self.actor = Actor(state_dim, action_dim, self.max_action)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim, self.max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        self.loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train(self):
        self.actor.train()# this is only actor - where is the critic?
        for epoch in range(1, self.epochs + 1):
            running = 0.0
            for xb, yb in self.loader:
                pred = self.actor(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += float(loss.item()) * xb.size(0)
            epoch_loss = running / len(self.loader.dataset)
            rospy.loginfo(f"[BC] Epoch {epoch}/{self.epochs} - MSE: {epoch_loss:.6f}")

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': optim.Adam(self.actor.parameters(), lr=3e-4).state_dict(),
            'critic_optimizer_state_dict': optim.Adam(self.critic.parameters(), lr=3e-4).state_dict(),
            'epsilon': 1.0,
            'total_it': 0,
        }, self.model_path)
        rospy.loginfo(f"Saved pretrained model to {self.model_path}")


if __name__ == '__main__':
    try:
        trainer = BehaviorCloningTrainer()
        trainer.train()
        trainer.save()
    except Exception as e:
        rospy.logerr(f"BC pretrain failed: {e}")


