from falcon9_env import Falcon9LandingEnv

# Create environment
env = Falcon9LandingEnv(render_mode="human")

# Reset and get initial observation
obs, info = env.reset()
higest=-999
# Take random actions
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if reward>higest:
        higest=reward
    if terminated or truncated:
        obs, info = env.reset()
    print(f'reward:{reward},terminated={terminated},truncated: {truncated}')
    print('')
    # if step%150==0:
    #     env.reset()

print(f'Highest:{higest}')
print('')
env.close()