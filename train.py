from stable_baselines3 import PPO
from falcon9_env import Falcon9LandingEnv
import torch
import time
# Your training code
def main():
    if torch.cuda.is_available():
        print("Using GPU !!!!!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print(" error :( . no gpu")
        return
    env = Falcon9LandingEnv(render_mode=None)
    model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./logs/", device="auto")
    
    model.learn(total_timesteps=1000000,progress_bar=True,tb_log_name="ppo_model_7.1_3")
    model.save('ppo_changed_metrics_1M_timesteps')
    print('model done .')
    env.close()

if __name__=="__main__":
    main()



# model=PPO.load('updated_reward_model_v2.zip',env,verbose=1,tensorboard_log="./logs/",device="cuda")
# model.learn(total_timesteps=100000, reset_num_timesteps=False, progress_bar=True,tb_log_name="ppo_model_6.9_2")
# print(" Model saved!")
# model.save("updated_reward_model_v2.zip")





