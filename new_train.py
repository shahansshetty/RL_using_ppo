from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
    BaseCallback
)
from lander_env import Falcon9LandingEnv
import torch
import os
import numpy as np

# Check GPU availability
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ No GPU detected, using CPU")
    device = "cpu"

# ============================================================================
# CRITICAL FIX #1: Add seed parameter and wrap with Monitor
# ============================================================================
def make_env(rank, seed=0):
    """
    Create environment with proper seeding and monitoring.
    Each environment gets a different seed for diversity.
    """
    def _init():
        env = Falcon9LandingEnv(render_mode=None)
        env = Monitor(env)  # CRITICAL: Wrap with Monitor for logging
        env.reset(seed=seed + rank)  # Different seed per environment
        return env
    return _init

# ============================================================================
# CRITICAL FIX #2: Custom callback to save VecNormalize with model
# ============================================================================
class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback to save VecNormalize statistics alongside model checkpoints.
    """
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            vecnorm_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps_vecnorm.pkl")
            
            self.model.save(model_path)
            
            # Save VecNormalize stats
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(vecnorm_path)
                if self.verbose > 0:
                    print(f"Saved model and VecNormalize at {self.n_calls} steps")
        
        return True

# ============================================================================
# MAIN TRAINING
# ============================================================================
if __name__ == '__main__':
    
    # Configuration
    n_envs = 16
    total_timesteps = 10_000_000  # Increased for better learning
    save_freq = 100_000  # Save every 100k steps
    eval_freq = 50_000   # Evaluate every 50k steps
    
    # Create directories
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./new_logs/", exist_ok=True)
    os.makedirs("./best_model/", exist_ok=True)
    
    print(f"🚀 Starting training with {n_envs} parallel environments")
    
    # ============================================================================
    # CRITICAL FIX #3: Create environments with proper seeding
    # ============================================================================
    env = SubprocVecEnv([make_env(i, seed=42) for i in range(n_envs)])
    
    # ============================================================================
    # CRITICAL FIX #4: Wrap with VecNormalize (ESSENTIAL!)
    # ============================================================================
    env = VecNormalize(
        env,
        norm_obs=True,        # Normalize observations
        norm_reward=True,     # Normalize rewards
        clip_obs=10.0,        # Clip normalized obs to [-10, 10]
        clip_reward=10.0,     # Clip normalized rewards to [-10, 10]
        gamma=0.995,          # Same as PPO gamma
    )
    
    print("✅ VecNormalize wrapper applied")
    
    # ============================================================================
    # CRITICAL FIX #5: Create separate evaluation environment
    # ============================================================================
    eval_env = SubprocVecEnv([make_env(i, seed=1000 + i) for i in range(4)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,    # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False,       # Don't update normalization stats during eval
        gamma=0.995,
    )
    
    print("✅ Evaluation environment created")
    
    # ============================================================================
    # CRITICAL FIX #6: Setup callbacks for monitoring and saving
    # ============================================================================
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/",
        name_prefix="falcon9_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,  # This should save VecNormalize automatically
    )
    
    # Custom callback to ensure VecNormalize is saved
    save_vecnorm_callback = SaveVecNormalizeCallback(
        save_freq=save_freq,
        save_path="./models/",
        name_prefix="falcon9_ppo",
        verbose=1
    )
    
    # Evaluation callback - evaluates agent and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_3/",
        log_path="./logs/",
        eval_freq=eval_freq,  # Evaluate every 50k steps (across all envs)
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # Combine callbacks
    callbacks = CallbackList([
        checkpoint_callback,
        save_vecnorm_callback,
        eval_callback
    ])
    
    print("✅ Callbacks configured")
    
    # ============================================================================
    # CRITICAL FIX #7: Improved hyperparameters
    # ============================================================================
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,         # Steps per env before update
        batch_size=512,       # Minibatch size
        n_epochs=10,          # Number of epochs per update
        gamma=0.995,          # Higher discount for long episodes
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,       # Entropy coefficient for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Actor network
                vf=[256, 256]   # Critic network
            ),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        tensorboard_log="./logs_new/",
        device=device,
    )
    
    print("✅ PPO model created")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Network architecture: [256, 256]")
    print(f"   - Device: {device}")


# Reloading the trained ppo model
    # model = PPO.load(
    #     "models/PPO_final_1",
    #     env=env,
    #     device=device,
    #     verbose=1,
    #     tensorboard_log="./logs/"
    # )
    # env = VecNormalize.load("models/ppo_final_vecnorm.pkl", env)
    # env.training = True
    # env.norm_reward = True




    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Steps per environment: {total_timesteps // n_envs:,}")
    print(f"Approximate training time: {total_timesteps // (n_envs * 60):.0f} minutes")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="falcon9_training",
            reset_num_timesteps=False  # Keep this False if continuing training
        )
        
        print("\n Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    
    except Exception as e:
        print(f"\n Training error: {e}")
        raise
    
    finally:
        # ============================================================================
        # CRITICAL FIX #8: Always save final model and VecNormalize
        # ============================================================================
        print("\n Saving final model...")
        
        final_model_path = "models/PPO_final_3.5"
        final_vecnorm_path = "models/ppo_final_vecnorm_3.5.pkl"
        
        model.save(final_model_path)
        env.save(final_vecnorm_path)
        
        print(f"✅ Final model saved to: {final_model_path}.zip")
        print(f"✅ VecNormalize saved to: {final_vecnorm_path}")
        
        # Close environments
        env.close()
        eval_env.close()
        
        print("\n🎉 Training session complete!")
        print("\nTo visualize training progress:")
        print("   tensorboard --logdir ./logs/")
        print("\nTo test the trained model, use:")
        print(f"   model = PPO.load('{final_model_path}')")
        print(f"   env = VecNormalize.load('{final_vecnorm_path}', test_env)")