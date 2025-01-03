"""
Just the PPO example from the lecture adapted to our Hockey env, it works pretty good and is fast but only works for discretized version.
"""



import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import optparse
import pickle
import os
import hockey.hockey_env as h_env
import matplotlib.pyplot as plt


os.makedirs('./results', exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

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

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluate current policy actions and values
            action_logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            # Compute advantages: advantage = rewards - V(s)
            advantages = rewards - state_values.detach()

            # Compute both clipped and unclipped objectives
            objective = ratios * advantages
            objective_clipped = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

            # Final loss: 
            # - We take the minimum of clipped/unclipped objective for the policy loss
            # - Add value function loss (using MSE)
            # - Subtract a small term for entropy to encourage exploration
            loss = -torch.min(objective, objective_clipped) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    # TODO: Consider adding suitable cmd line arguments

    ############## Hyperparameters ##############
    env_name = 'hockey'
    # creating environment
    env = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv.NORMAL, weak_opponent=False)
    state_dim = env.observation_space.shape[0]
    action_dim = 8
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 20000    # max training episodes
    max_timesteps = 1000         # max timesteps in one episode
    n_latent_var = 128           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 10               # update policy for K epochs
    eps_clip = 0.2   # clip parameter for PPO
    random_seed = 123
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(env_name,"Clipping:", eps_clip)

    # logging variables
    rewards = []
    lengths = []
    timestep = 0

    def save_statistics():
        with open(f"./results/PPO_{env_name}-eps{eps_clip}-seed{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards": rewards, "lengths": lengths,
                     "eps": eps_clip, "seed": random_seed},
                    f)

            


    # training loop
    for i_episode in range(1, max_episodes+1):
        state, info = env.reset()

        running_reward=0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            a1_discrete = ppo.policy_old.act(state, memory)
            a1 = env.discrete_to_continous_action(a1_discrete)
            state, reward, done, _trunc, _ = env.step(a1) 
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if done:
                break
            #env.render()

        rewards.append(running_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(ppo.policy.state_dict(), f'./results/PPO_{env_name}_{i_episode}-eps{eps_clip}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            # stop training if avg_reward > solved_reward
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
            if avg_reward > solved_reward:
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), f'./results/PPO_{env_name}-eps{eps_clip}-k{K_epochs}-solved.pth')
    
    plt.plot(rewards, marker='o', linestyle='-', color='b', label='Score History')
    plt.title('Score History')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
    save_statistics()

if __name__ == '__main__':
    main()