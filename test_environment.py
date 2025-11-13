from lander_env import Falcon9LandingEnv
import time
# Create environment
env = Falcon9LandingEnv(render_mode="human")

# Reset and get initial observation
obs, info = env.reset()
highest,lowest=-999,999
l_count,count=0,0
# Take random actions
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info= env.step(action)
    if reward>highest:
        highest=reward
    if reward<lowest:
        lowest=reward
    
        
    if info['landed_successfully']:
        time.sleep(5)
        print(f'reward:{reward},terminated={terminated},truncated: {truncated},landed:{info['landed_successfully']}')
        print('')
        print(f'Highest:{highest}')
        print('')
        l_count+=1
        env.close()
        
    if terminated or truncated:
        # time.sleep(5)
        count+=1
        obs, info = env.reset()
    print(f'reward:{reward},terminated={terminated},truncated: {truncated}')
    print('')
    
    # if step%150==0:
    #     env.reset()

print(f'Highest:{highest},lowest: {lowest},landing_count : {l_count},total_eps: {count}')

print('')
env.close()