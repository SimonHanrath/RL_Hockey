"""
Test script for PPO agent. Is kinda bad as we need to redefine the same model here. This can easily be fixed by updating the save logic as in the SAC example TODO!

"""


import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import sys
import os
import hockey.hockey_env as h_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

def render_policy(checkpoint_path, num_episodes=5, max_timesteps=300):
    # Load environment
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)  
    state_dim = env.observation_space.shape[0]
    action_dim = 8  # from your training script
    n_latent_var = 256

    # Load policy
    policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    # Run a few episodes
    for episode in range(num_episodes):
        state, _info = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            # Render the environment
            env.render()

            # Get action from the policy
            discrete_action = policy.act(state)
            continuous_action = env.discrete_to_continous_action(discrete_action)
            #opponent_action = [0,0.,0,0]  # assuming a fixed opponent

            # Step environment
            next_state, reward, done, _trunc, _info = env.step(continuous_action)

            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode {episode+1} finished after {t+1} timesteps. Reward: {episode_reward}")
                break
    
    env.close()

if __name__ == "__main__":
    # Example usage:
    # python render_checkpoint.py ./results/PPO_LunarLander-v3_500-eps0.2.pth
    # Or modify the line below to point to your chosen checkpoint.
   
    num_episodes = 10
    checkpoint_file = 'results/PPO_LunarLander-v3_20000-eps0.2.pth'

    render_policy(checkpoint_file, num_episodes=num_episodes)
