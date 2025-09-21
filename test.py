from stable_baselines3 import PPO
from lander_env import Falcon9LandingEnv

# Load your trained model
model = PPO.load("PPO_with_5_AP_2.zip")
# Create environment with rendering to watch
env = Falcon9LandingEnv(render_mode="human")

# Use the model
obs, info = env.reset()

highest=-999
l_count,count=0,0

for step in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        if info["landed_successfully"]:
            print(f"Landing completed!")
            break
        else:
            print('crashed')
        print(f"Distance from target: {info['distance_to_target'] :.2f}m")
        print(f"Landing speed: {info['speed']:.2f}m/s")
        count+=1
        obs, info = env.reset()

    if reward>highest:
        highest=reward

    if reward==100:
        l_count+=1
    # if step%200==0:
    #     print(step)
    


print(f'Highest:{highest},landing_count : {l_count},total_eps: {count}')
print('')
env.close()