import gym
import torch
from agent import Agent
import hockey.hockey_env as h_env
import numpy as np


if __name__ == '__main__':
    # Initialize the Lunar Lander environment
    env = h_env.HockeyEnv()

    # Initialize the agent with the same parameters as during training
    agent = Agent(
    gamma=0.95,                  
    epsilon=0,                   
    batch_size=128,              
    n_actions=8,                 
    eps_end=0.01,                
    input_dims=[18],             
    lr = 0.0005,                    
    max_mem_size=100000,         
    eps_dec=(1 - 0.01) / 50000, 
    replace = 500  
)

    # Load the trained model weights
    agent.Q_eval.load_state_dict(torch.load('DQN/model_weights/DQN_player_2000_episodes.pth'))
    agent.Q_eval.eval()  # Set the model to evaluation mode

    # Run the environment
    n_games = 5  # Number of episodes to render
    for i in range(n_games):
        observation = env.reset()[0]
        
        done = False
        score = 0

        while not done:
            env.render()  # Render the environment
            a1_discrete = agent.choose_action(observation)
            a1 = env.discrete_to_continous_action(a1_discrete)
            a2 = [0,0,0,0] 
            action = np.hstack([a1,a2])
            observation_, reward, done, _trunc, _ = env.step(action)
            score += reward
            observation = observation_

        print(f'Episode {i+1} Score: {score}')
    
    env.close()  # Close the rendering window
