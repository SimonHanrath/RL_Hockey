"""
PPO self train script
"""

import sys
import os
import argparse
sys.path.append(os.getcwd())

import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os
import hockey.hockey_env as h_env
import model as model
import helper_functions as functions
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


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






class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, network_type, original_model_path):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        if network_type == "tanh_1_layer":
            self.policy = model.ActorCritic_single_hidden(state_dim, action_dim, n_latent_var).to(device)      
            self.policy_old = model.ActorCritic_single_hidden(state_dim, action_dim, n_latent_var).to(device)
        elif network_type == "relu_2_layer":
            self.policy = model.ActorCritic_relu(state_dim, action_dim, n_latent_var).to(device)      
            self.policy_old = model.ActorCritic_relu(state_dim, action_dim, n_latent_var).to(device)
        else:
            self.policy = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)      
            self.policy_old = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.load_state_dict(torch.load(original_model_path, map_location=device))
        self.policy.eval()

        self.policy_old.load_state_dict(torch.load(original_model_path, map_location=device))
        self.policy_old.eval()

        self.MseLoss = nn.MSELoss()

    def update(self, memory, entropy_factor):
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

        dataset = TensorDataset(old_states, old_actions, old_logprobs, rewards)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        for _ in range(self.K_epochs):
            for batch_states, batch_actions, batch_logprobs, batch_rewards in dataloader:
                # Evaluate current policy
                action_logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)

                # Compute advantages
                advantages = batch_rewards - state_values.detach()

                # Compute PPO ratio
                ratios = torch.exp(action_logprobs - batch_logprobs.detach())

                # Compute clipped and unclipped PPO objectives
                objective = ratios * advantages
                objective_clipped = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                # Compute final loss
                loss = -torch.min(objective, objective_clipped) + 0.5 * self.MseLoss(state_values, batch_rewards) - entropy_factor * dist_entropy

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()



        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--own_reward', type=int, default=1, help='0 or 1')
    parser.add_argument('--checkpoint_path', type=str, default="PPO/selfplay_history/checkpoints")
    parser.add_argument('--original_model_path', type=str, default="PPO/checkpoints/gamma_0.96/best_PPO_hockey-eps0.2.pth")
    parser.add_argument('--use_large_action_space', type=int, default= 1)
    parser.add_argument('--use_entropy_decay', type = int, default= 0)
    parser.add_argument('--num_updates', type = int, default= 10000)
    parser.add_argument('--n_latent_var', type = int, default= 256)
    parser.add_argument('--lr', type = float, default= 0.0003)
    parser.add_argument('--gamma', type = float, default= 0.96)
    parser.add_argument('--start_entropy', type = float, default= 0.05)
    parser.add_argument('--end_entropy', type = float, default= 0.005)
    parser.add_argument('--epochs_per_update', type = int, default= 4)
    parser.add_argument('--later_sparse_rewards', type = int, default= 0)
    parser.add_argument('--updates_between_renders', type = int, default= 0)
    parser.add_argument('--network_type', type = str, default= "tanh_2_layer")
    parser.add_argument('--n_net_memory', type = int, default= 5)




    args = parser.parse_args()

    writer = SummaryWriter('PPO/tensorboard/' + os.path.basename(args.checkpoint_path))

    use_own_reward = bool(args.own_reward)
    checkpoint_path = args.checkpoint_path
    use_large_action_space = bool(args.use_large_action_space)
    use_entropy_decay = bool(args.use_entropy_decay)
    later_sparse_rewards = bool(args.later_sparse_rewards)
    updates_between_renders = args.updates_between_renders
    network_type = args.network_type
    original_model_path = args.original_model_path
    n_net_memory = args.n_net_memory

    ############## Hyperparameters ##############
    env_name = 'hockey'
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)
    state_dim = env.observation_space.shape[0]
    if use_large_action_space:
        action_dim = 28
    else:
        action_dim = 8

    num_updates = args.num_updates
    updates_between_logs = 1           # print avg reward in the interval
    max_timesteps = 1000         # max timesteps in one episode
    n_latent_var = args.n_latent_var           # number of variables in hidden layer
    updates_between_saves = 200
    updates_between_match_histories = 10
    lr = args.lr 
    betas = (0.9, 0.999)
    gamma = args.gamma              # discount factor
    K_epochs = args.epochs_per_update               # update policy for K epochs
    eps_clip = 0.2   # clip parameter for PPO
    start_entropy = args.start_entropy
    end_entropy = args.end_entropy
    standard_entropy = 0.01
    selfplay_epochs = 500

    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, network_type, original_model_path)
    print(env_name,"Clipping:", eps_clip)
    best_winrate = 0

    # logging variables
    rewards = []
    lengths = []

    episode_match_history = [0, 0, 0]
    sparse_rewards = False

    

    original_policy = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)
    original_policy.load_state_dict(torch.load(original_model_path, map_location=device))
    original_policy.eval()

    previous_policies = [original_policy]

    # training loop
    for epoch in range(selfplay_epochs):
        update_id = 0
        while True:
            update_id += 1
            if update_id > num_updates/2 and later_sparse_rewards:
                sparse_rewards = True # update_id > num_updates/2
            running_reward=0
            running_length=0
            frames = 0
            games_played = 0
            while frames < 2500:
                games_played+=1
                state, info = env.reset()

                if env.puck.position[0] < h_env.CENTER_X:
                    puck_crossed_line = False
                else:
                    puck_crossed_line = True   

                policy_index = random.randint(1,len(previous_policies))-1
                #print(policy_index)
                enemy_policy = previous_policies[policy_index] 
                
                for t in range(max_timesteps):
                    if (frames<500) and not(updates_between_renders == 0) and update_id%updates_between_renders == 0:
                        env.render()
                    frames += 1

                    # Running policy_old:
                    a1_discrete = ppo.policy_old.act(state, memory)
 
                    a2_discrete = enemy_policy.act(env.obs_agent_two())

                    if use_large_action_space:
                        a1 = functions.large_discrete_to_continous_action(env, a1_discrete)
                        a2 = functions.large_discrete_to_continous_action(env, a2_discrete)
                    else:
                        a1 = env.discrete_to_continous_action(a1_discrete)
                        a2 = env.discrete_to_continous_action(a2_discrete)

                    old_puck_x = env.puck.position[0]

                    combined_action = np.hstack([a1, a2])

                    state, reward, done, _trunc, info = env.step(combined_action) 
                    if (old_puck_x < h_env.CENTER_X) and env.puck.position[0] > h_env.CENTER_X:
                        puck_crossed_line = True
                    elif env.player1_has_puck == h_env.MAX_TIME_KEEP_PUCK:
                        puck_crossed_line = False

                    if use_own_reward:
                        reward = functions.custom_reward(env, puck_crossed_line, info, sparse_rewards)

                    # Saving reward and is_terminal:
                    memory.rewards.append(reward)
                    memory.is_terminals.append(done)

                    running_reward += reward
                    

                    if done:
                        running_length += t+1
                        if env.winner == 0:  # tie
                            episode_match_history[1] += 1
                        elif env.winner == 1:  # you won
                            episode_match_history[0] += 1
                        else:  # opponent won
                            episode_match_history[2] += 1
                        break

            #implement entropy decay to reduce exploration over time
            if use_entropy_decay:
                k = math.log(end_entropy/ start_entropy) / num_updates
                entropy_factor =  start_entropy * math.exp(k * update_id)
            else:
                entropy_factor = standard_entropy        

            ppo.update(memory, entropy_factor)
            memory.clear_memory()
            rewards.append(running_reward)
            lengths.append(running_length)

            winrate = episode_match_history[0]/(episode_match_history[0]+episode_match_history[2])
            avg_reward = np.mean(rewards[-updates_between_logs:])/games_played
            avg_length = int(np.mean(lengths[-updates_between_logs:])/games_played)

            # logging
            if update_id % updates_between_logs == 0:
                print('Update {} \t avg length: {} \t avg game reward: {}'.format(update_id, avg_length, avg_reward))

            if update_id % updates_between_match_histories == 0:
                print("Match history" + str(episode_match_history) + " with winrate " + str(winrate))
                if winrate > 0.55:
                    print("########## Saving checkpoint version" + str(epoch) + "... ##########")
                    filename = f'/PPO_selfplay' + str(epoch) + '.pth'
                    torch.save(ppo.policy.state_dict(), f'./' + checkpoint_path + filename)
                    new_policy = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)
                    new_policy.load_state_dict(torch.load(checkpoint_path + filename, map_location=device))
                    new_policy.eval()
                    previous_policies.append(new_policy)
                    if len(previous_policies) > n_net_memory:
                        del previous_policies[0]
                    episode_match_history = [0, 0, 0]
                    break
                episode_match_history = [0, 0, 0]



    writer.close()

if __name__ == '__main__':
    main()