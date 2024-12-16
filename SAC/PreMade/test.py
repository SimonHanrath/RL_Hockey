import gym
from stable_baselines3 import SAC

# Create the environment
env = gym.make("Pendulum-v1", render_mode='human')

# Load the previously saved model
model = SAC.load("sac_pendulum", env=env)

# Now you can use the model to predict actions, evaluate, etc.
obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action, _states)
    observation_, reward, done, truncated, info = env.step(action)
    done = done or truncated
    obs=observation_
    env.render()
    if done:
        obs, info = env.reset()

env.close()
