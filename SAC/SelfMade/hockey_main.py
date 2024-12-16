import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env



from torch.utils.tensorboard import SummaryWriter
import time


if __name__ == '__main__':
    #env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)  

    # this determines whether we train a new model (False) or let an existing one play (True)
    test_mode = False

    # Initialize the agent TODO: currently we set most params of the agent implicetly in the agent or the model. I want it all explicetly here
    agent = Agent(input_dims=env.observation_space.shape, env=env, 
                  n_actions=4)
    
    n_games = 20000 # number of games played during training
    best_score = -np.inf
    score_history = []
    avg_score_history = []

    # Initialize TensorBoard writer (access: tensorboard --logdir=runs in terminal)
    writer = SummaryWriter('runs/hockey_sac_training')

    if test_mode:
        agent.load_models() # TODO: change this to accept directory and not just have default
        
        observation = env.reset()[0]
        env.render()
        for i in range(n_games):  
            observation = env.reset()[0]
            done = False
            score = 0
            while not done:
                time.sleep(0.1) # I am old and I need this to understand what is happening lol
                env.render()
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                done = done or truncated
                score += reward
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
            if avg_score > best_score:
                best_score = avg_score
            print(f'episode {i}, score {score:.2f}, avg score {avg_score:.2f}')
        env.close()
    else:
        for i in range(n_games):
            observation = env.reset()[0]
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                done = done or truncated
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_

            score_history.append(score)
            avg_score = np.mean(score_history[-1000:])
            avg_score_history.append(avg_score)

            # TensorBoard logging
            writer.add_scalar('Rewards/Episode_Score', score, i)
            writer.add_scalar('Rewards/Avg_Score', avg_score, i)

            # Save model if best score
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print(f'episode {i}, score {score:.2f}, avg score {avg_score:.2f}')

        writer.close()
        print("Training completed. Logs saved to TensorBoard.")
