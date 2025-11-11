#!/usr/bin/env python3

import numpy as np
import glob
import os

demos_dir = "/home/user/turtlebot3_ws/src/td3_rl_controller_high_buffer_size_respawn/demos"
files = sorted(glob.glob(os.path.join(demos_dir, '*.npz')))

print("=" * 70)
print("DEMO DATA ANALYSIS")
print("=" * 70)

all_actions = []

for i, f in enumerate(files, 1):
    data = np.load(f)
    states = data['states']
    actions = data['actions']
    timestamps = data['timestamps']

    linear_vel = actions[:, 0]  # linear.x
    angular_vel = actions[:, 1]  # angular.z

    print(f"\nDemo {i}: {os.path.basename(f)}")
    print(f"  Samples: {len(states)}")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.1f}s")
    print(f"  Linear velocity (v):")
    print(f"    Mean: {linear_vel.mean():.4f}, Std: {linear_vel.std():.4f}")
    print(f"    Min: {linear_vel.min():.4f}, Max: {linear_vel.max():.4f}")
    print(f"    Forward (v>0.01): {(linear_vel > 0.01).sum()} samples ({100*(linear_vel > 0.01).sum()/len(linear_vel):.1f}%)")
    print(f"    Backward (v<-0.01): {(linear_vel < -0.01).sum()} samples ({100*(linear_vel < -0.01).sum()/len(linear_vel):.1f}%)")
    print(f"    Stopped (|v|<0.01): {(np.abs(linear_vel) < 0.01).sum()} samples ({100*(np.abs(linear_vel) < 0.01).sum()/len(linear_vel):.1f}%)")
    print(f"  Angular velocity (w):")
    print(f"    Mean: {angular_vel.mean():.4f}, Std: {angular_vel.std():.4f}")
    print(f"    Min: {angular_vel.min():.4f}, Max: {angular_vel.max():.4f}")
    print(f"    Left turn (w>0.1): {(angular_vel > 0.1).sum()} samples ({100*(angular_vel > 0.1).sum()/len(angular_vel):.1f}%)")
    print(f"    Right turn (w<-0.1): {(angular_vel < -0.1).sum()} samples ({100*(angular_vel < -0.1).sum()/len(angular_vel):.1f}%)")

    all_actions.append(actions)

# Combined analysis
all_actions = np.concatenate(all_actions, axis=0)
linear_all = all_actions[:, 0]
angular_all = all_actions[:, 1]

print("\n" + "=" * 70)
print("COMBINED DATASET ANALYSIS")
print("=" * 70)
print(f"Total samples: {len(all_actions)}")
print(f"\nLinear velocity distribution:")
print(f"  Mean: {linear_all.mean():.4f}")
print(f"  Forward motion: {(linear_all > 0.01).sum()} samples ({100*(linear_all > 0.01).sum()/len(linear_all):.1f}%)")
print(f"  Backward motion: {(linear_all < -0.01).sum()} samples ({100*(linear_all < -0.01).sum()/len(linear_all):.1f}%)")
print(f"  Stopped: {(np.abs(linear_all) < 0.01).sum()} samples ({100*(np.abs(linear_all) < 0.01).sum()/len(linear_all):.1f}%)")

print(f"\nAngular velocity distribution:")
print(f"  Mean: {angular_all.mean():.4f}")
print(f"  Turning: {(np.abs(angular_all) > 0.1).sum()} samples ({100*(np.abs(angular_all) > 0.1).sum()/len(angular_all):.1f}%)")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if (linear_all < -0.01).sum() / len(linear_all) > 0.5:
    print("⚠️  WARNING: Over 50% of actions are BACKWARD movement!")
    print("   Your demos might be mostly reversing, not 3-point turns.")

if (np.abs(angular_all) < 0.1).sum() / len(angular_all) > 0.7:
    print("⚠️  WARNING: Over 70% of actions have low angular velocity!")
    print("   3-point turns should have significant turning motion.")

if (linear_all > 0.01).sum() / len(linear_all) < 0.2:
    print("⚠️  WARNING: Less than 20% forward motion!")
    print("   3-point turns need forward movement too.")

print("\n" + "=" * 70)
