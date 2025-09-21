from stable_baselines3 import PPO
from lander_env import Falcon9LandingEnv
import torch
import time
# Your training code

# if torch.cuda.is_available():
#     print("Using GPU !!!!!")
#     print(f"Device Name: {torch.cuda.get_device_name(0)}")
# else:
#     print(" error :( . no gpu")
    # return



env = Falcon9LandingEnv(render_mode=None)
model = PPO("MlpPolicy",env,tensorboard_log="./logs/"
)
model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./logs/",learning_rate=0.00003,ent_coef=0.01 )
model.learn(total_timesteps=1000000,progress_bar=True,tb_log_name="PPO_with_5_AP_5",)
model.save('PPO_with_5_AP_5')

# model=PPO.load('PPO_with_5_AP_2.zip',env,verbose=1,tensorboard_log="./logs/")
# model.learn(total_timesteps=1000000, reset_num_timesteps=False, progress_bar=True,tb_log_name="PPO_with_5_AP_2")
# print(" Model saved!")
# model.save("PPO_with_5_AP_2.zip")

print('model done .')
env.close()




