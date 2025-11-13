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
import time

# Check GPU
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ No GPU detected, using CPU")
    device = "cpu"

# ============================================================================
# CURRICULUM LEARNING CALLBACK
# ============================================================================
class CurriculumCallback(BaseCallback):
    """
    Automatically increases curriculum level when agent performs well.
    This is the key to making the agent learn progressively!
    """
    def __init__(self, eval_env, check_freq=50000, success_threshold=0.7, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.success_threshold = success_threshold
        self.current_level = 1
        self.evaluations_since_upgrade = 0
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate current performance
            n_eval_episodes = 20
            success_count = 0
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"🔍 Curriculum Evaluation at {self.n_calls} steps (Level {self.current_level})")
            
            for episode in range(n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    if done:
                        # Check if it was a successful landing
                        if isinstance(info, list):
                            info = info[0]
                        if info.get('landed_successfully', False):
                            success_count += 1
            
            success_rate = success_count / n_eval_episodes
            
            if self.verbose > 0:
                print(f"✅ Success Rate: {success_rate:.1%} ({success_count}/{n_eval_episodes})")
            
            # Check if we should advance curriculum
            if success_rate >= self.success_threshold and self.current_level < 5:
                self.current_level += 1
                self.evaluations_since_upgrade = 0
                
                # Update all training environments to new level
                try:
                    # Get the base VecEnv from VecNormalize
                    base_env = self.training_env.venv if hasattr(self.training_env, 'venv') else self.training_env
                    
                    # Update each sub-environment
                    for env_idx in range(base_env.num_envs):
                        base_env.env_method('set_curriculum_level', self.current_level, indices=[env_idx])
                    
                    # Update eval environment too
                    eval_base = self.eval_env.venv if hasattr(self.eval_env, 'venv') else self.eval_env
                    for env_idx in range(eval_base.num_envs):
                        eval_base.env_method('set_curriculum_level', self.current_level, indices=[env_idx])
                    
                    if self.verbose > 0:
                        print(f"🎓 CURRICULUM ADVANCED! Now at Level {self.current_level}/5")
                        print(f"   New difficulty: Higher altitude, smaller landing zone, stricter requirements")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"⚠️ Warning: Could not update curriculum level: {e}")
            else:
                self.evaluations_since_upgrade += 1
                if self.verbose > 0:
                    if self.current_level < 5:
                        print(f"📚 Need {self.success_threshold:.0%} success rate to advance")
                    else:
                        print(f"🏆 Maximum curriculum level reached!")
            
            if self.verbose > 0:
                print(f"{'='*60}\n")
        
        return True

# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================
def make_env(rank, seed=0, curriculum_level=1):
    """Create environment with curriculum support"""
    def _init():
        env = Falcon9LandingEnv(
            render_mode=None, 
            difficulty='easy',
            curriculum_level=curriculum_level
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

# ============================================================================
# SAVE CALLBACK
# ============================================================================
class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize with checkpoints"""
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
            
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(vecnorm_path)
                if self.verbose > 0:
                    print(f"💾 Saved checkpoint at {self.n_calls} steps")
        
        return True

# ============================================================================
# MAIN TRAINING
# ============================================================================
if __name__ == '__main__':
    
    # Configuration
    n_envs = 16  # Parallel environments
    total_timesteps = 10_000_000  # 10M steps
    save_freq = 500_000  # Save every 500k
    eval_freq = 100_000  # Evaluate every 100k
    curriculum_check_freq = 50000  # Check curriculum every 50k
    
    # Create directories
    os.makedirs("./models_curriculum/", exist_ok=True)
    os.makedirs("./logs_curriculum/", exist_ok=True)
    os.makedirs("./best_model_curriculum/", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 FALCON 9 LANDING - CURRICULUM LEARNING")
    print(f"{'='*70}")
    print(f"Training Configuration:")
    print(f"  - Parallel Environments: {n_envs}")
    print(f"  - Total Training Steps: {total_timesteps:,}")
    print(f"  - Device: {device}")
    print(f"  - Curriculum Levels: 5 (automatic progression)")
    print(f"  - Success Threshold for Advancement: 70%")
    print(f"{'='*70}\n")
    
    # Create training environments (all start at level 1)
    print("🔧 Creating training environments...")
    env = SubprocVecEnv([make_env(i, seed=42, curriculum_level=1) for i in range(n_envs)])
    
    # CRITICAL: Adjust VecNormalize for new reward scale
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,  # Rewards are now -10 to +100, so this is fine
        gamma=0.99,
    )
    
    print("✅ Training environment created with VecNormalize")
    
    # Create evaluation environment
    print("🔧 Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(0, seed=9999, curriculum_level=1)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize during eval
        clip_obs=10.0,
        training=False,
        gamma=0.99,
    )
    
    print("✅ Evaluation environment created")
    
    # Setup callbacks
    print("🔧 Setting up callbacks...")
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models_curriculum/",
        name_prefix="falcon9_curriculum",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # VecNormalize save callback
    save_vecnorm_callback = SaveVecNormalizeCallback(
        save_freq=save_freq,
        save_path="./models_curriculum/",
        name_prefix="falcon9_curriculum",
        verbose=1
    )
    
    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_curriculum/",
        log_path="./logs_curriculum/",
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # CURRICULUM CALLBACK - This is the magic!
    curriculum_callback = CurriculumCallback(
        eval_env=eval_env,
        check_freq=curriculum_check_freq,
        success_threshold=0.7,  # Need 70% success to advance
        verbose=1
    )
    
    # Combine callbacks
    callbacks = CallbackList([
        checkpoint_callback,
        save_vecnorm_callback,
        eval_callback,
        curriculum_callback
    ])
    
    print("✅ Callbacks configured")
    
    # Create PPO model
    print("🔧 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        tensorboard_log="./logs_curriculum/",
        device=device,
    )
    
    print("✅ PPO model created")
    print(f"   Network: [256, 256] x 2")
    print(f"   Learning rate: 3e-4")
    print(f"   Device: {device}")
    
    print(f"\n{'='*70}")
    print(f"🎓 CURRICULUM LEARNING STRATEGY")
    print(f"{'='*70}")
    print(f"Level 1 (Current): Very Easy")
    print(f"  - Altitude: 10-15m | Zone: 15m | Speed: <10 m/s | Tilt: <32°")
    print(f"\nLevel 2: Easy")
    print(f"  - Altitude: 15-20m | Zone: 12m | Speed: <8 m/s | Tilt: <26°")
    print(f"\nLevel 3: Medium")
    print(f"  - Altitude: 20-30m | Zone: 10m | Speed: <6 m/s | Tilt: <21°")
    print(f"\nLevel 4: Hard")
    print(f"  - Altitude: 30-40m | Zone: 8m | Speed: <5 m/s | Tilt: <18°")
    print(f"\nLevel 5: Expert")
    print(f"  - Altitude: 40-50m | Zone: 7m | Speed: <4 m/s | Tilt: <14°")
    print(f"\n⚡ Agent will automatically advance when achieving 70% success rate!")
    print(f"{'='*70}\n")
    
    input("Press Enter to start training...")
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Expected training time: ~{total_timesteps // (n_envs * 60):.0f} minutes")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="falcon9_curriculum",
            reset_num_timesteps=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"⏱️  Trained for: {elapsed_time/60:.1f} minutes")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        print("\n💾 Saving final model...")
        
        final_model_path = "models_curriculum/falcon9_final"
        final_vecnorm_path = "models_curriculum/falcon9_final_vecnorm.pkl"
        
        model.save(final_model_path)
        env.save(final_vecnorm_path)
        
        print(f"✅ Final model: {final_model_path}.zip")
        print(f"✅ VecNormalize: {final_vecnorm_path}")
        
        env.close()
        eval_env.close()
        
        print(f"\n{'='*70}")
        print(f"🎉 TRAINING SESSION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 To visualize training:")
        print(f"   tensorboard --logdir ./logs_curriculum/")
        print(f"\n🧪 To test the model:")
        print(f"   python test_model.py")
        print(f"\n📁 Models saved in:")
        print(f"   - ./models_curriculum/ (checkpoints)")
        print(f"   - ./best_model_curriculum/ (best model)")
        print(f"{'='*70}\n")