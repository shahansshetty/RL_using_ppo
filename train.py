from stable_baselines3 import PPO
from falcon9_env import Falcon9LandingEnv

# Your training code
env = Falcon9LandingEnv(render_mode=None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# ADD THIS LINE TO SAVE:
model.save("falcon9_landing_ppo_v2")
print("✅ Model saved!")

env.close()