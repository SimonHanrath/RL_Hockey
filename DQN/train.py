import gym
from agent import Agent
import hockey.hockey_env as h_env
import numpy as np
import matplotlib.pyplot as plt
import torch

model_weights = None#'DQN/model_weights/DQN_player_1000_episodes.pth'

if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    agent = Agent(
    gamma=0.95,                  # Slightly lower gamma for faster learning
    epsilon=1,                   # Start with full exploration
    batch_size=128,              # Larger batch size for stability
    n_actions=8,                 # Matches the action space
    eps_end=0.01,                # Minimum epsilon
    input_dims=[18],             # Matches state observation space
    lr = 0.0005,                    # Reduced learning rate for stability
    max_mem_size=100000,         # Increased replay memory size
    eps_dec=(1 - 0.01) / 50000, # Decay epsilon over 100,000 steps
    replace = 500  # Update the target network every 500 steps
)
    
    if model_weights != None:
        agent.Q_eval.load_state_dict(torch.load(model_weights))
    
    scores, eps_history = [], []
    n_games = 2000

    for i in range(n_games):
        
        score = 0
        done = False
        observation, info = env.reset()
        
        while not done:
            a1_discrete = agent.choose_action(observation)
            a1 = env.discrete_to_continous_action(a1_discrete)
            a2 = [0,0,0,0] 
            action = np.hstack([a1,a2])
            observation_, reward, done, _trunc, _ = env.step(action)
            score += reward
            agent.store_transition(observation, a1_discrete, reward, observation_, done)

            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f'episode: {i} score: {score} average score {avg_score} epsilon {agent.epsilon}')

        # Save model weights every 100 episodes
        if (i + 1) % 100 == 0:
            torch.save(agent.Q_eval.state_dict(), f'DQN/model_weights/DQN_player_{i+1}_episodes.pth')
            print(f"Model weights saved after {i+1} episodes.")

    x = [i + 1 for i in range(len(scores))]
    plt.plot(x, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Performance of DQN on LunarLander-v2')
    plt.show()
