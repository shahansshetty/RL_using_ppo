# test_model.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from lander_env import Falcon9LandingEnv
import numpy as np
import time

def test_model(model_path, vecnorm_path, n_episodes=10):
    """Test trained model with rendering"""
    
    # Create test environment (single env with rendering)
    test_env = DummyVecEnv([lambda: Falcon9LandingEnv(render_mode="human")])
    
    # Load VecNormalize stats
    test_env = VecNormalize.load(vecnorm_path, test_env)
    test_env.training = False  # Don't update stats
    test_env.norm_reward = False  # See actual rewards
    
    # Load model
    model = PPO.load(model_path, env=test_env)
    
    print(f"Testing model for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    successful_landings = 0
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done:
                if info[0].get('landed_successfully', False):
                    successful_landings += 1
                    time.sleep(3)
                    print(f"✅ Episode {episode+1}: LANDED! "
                          f"Reward: {episode_reward:.0f}, "
                          f"Distance: {info[0]['distance_to_target']:.2f}m")
                else:
                    print(f"💥 Episode {episode+1}: CRASHED. "
                          f"Reward: {episode_reward:.0f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Success rate: {successful_landings}/{n_episodes} "
          f"({successful_landings/n_episodes*100:.1f}%)")
    print(f"  Average reward: {np.mean(episode_rewards):.0f} ± {np.std(episode_rewards):.0f}")
    print(f"  Average length: {np.mean(episode_lengths):.0f} steps")
    print(f"{'='*60}")
    
    test_env.close()

if __name__ == "__main__":
    # Test best model
    test_model(
        model_path='best_model_2/best_model',
        vecnorm_path="models/ppo_final_vecnorm_2.pkl",
        n_episodes=5
    )