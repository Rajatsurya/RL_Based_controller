#!/usr/bin/env python3

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  NumPy version: {np.__version__}")
        
        import rospy
        print("✓ ROS imported successfully")
        
        # Test TD3 components
        from td3_agent import TD3Agent, Actor, Critic
        print("✓ TD3 agent components imported successfully")
        
        from reward_function import RewardFunction
        print("✓ Reward function imported successfully")
        
        from statespace_vector import RLStateBuilder
        print("✓ State space vector builder imported successfully")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_networks():
    """Test if neural networks can be created"""
    try:
        import torch
        from td3_agent import Actor, Critic
        
        # Test actor network
        actor = Actor(state_dim=29, action_dim=2, max_action=1.0)
        test_state = torch.randn(1, 29)
        action = actor(test_state)
        print(f"✓ Actor network created successfully")
        print(f"  Input shape: {test_state.shape}")
        print(f"  Output shape: {action.shape}")
        print(f"  Output range: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
        # Test critic network
        critic = Critic(state_dim=29, action_dim=2)
        test_action = torch.randn(1, 2)
        q1, q2 = critic(test_state, test_action)
        print(f"✓ Critic network created successfully")
        print(f"  Q1 shape: {q1.shape}")
        print(f"  Q2 shape: {q2.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing TD3 RL Controller Components")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    # Test networks
    print("\n2. Testing neural networks...")
    networks_ok = test_networks()
    
    # Summary
    print("\n" + "=" * 40)
    if imports_ok and networks_ok:
        print("✓ All tests passed! TD3 system is ready to use.")
        print("\nTo start training, run:")
        print("  roslaunch td3_rl_controller td3_training.launch")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 