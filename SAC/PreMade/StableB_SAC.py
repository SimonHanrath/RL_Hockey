import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import hockey.hockey_env as h_env
import numpy as np


# Create the environment
env = gym.make("Pendulum-v1")

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("sac_pendulum")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

