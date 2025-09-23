from stable_baselines3 import PPO
from lander_env import Falcon9LandingEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

#for cuda

import torch
# import time
import sys
# Your training code

if torch.cuda.is_available():
    print("Using GPU !!!!!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print(" error :( . no gpu")
    sys.exit()

def make_env():
    return Falcon9LandingEnv(render_mode=None)



# 

# model=PPO.load('PPO_with_5_AP_6.zip',env,verbose=1,tensorboard_log="./new_logs/")
# model.learn(total_timesteps=500000, reset_num_timesteps=False, progress_bar=True,tb_log_name="PPO_with_5_AP_6")
# print(" Model saved!")
# model.save("PPO_with_5_AP_6-2.zip")

# print('model done .')
# env.close()

if __name__=='__main__':
    env = SubprocVecEnv([make_env for _ in range(8)])
    model = PPO("MlpPolicy",env,tensorboard_log="./new_logs/",device='cuda',batch_size=128,n_steps=512)
    # model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./logs/",learning_rate=0.00003,ent_coef=0.01 )
    model.learn(total_timesteps=1000000,progress_bar=True,tb_log_name="PPO_with_5_AP_7",)
    model.save('PPO_with_5_AP_7')
