import gym
import numpy as np
from SAC.SelfMade.agent.agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
import yaml
import os

# load config file for model and training specifications
with open(os.path.join('SAC', 'SelfMade', 'train', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)


def get_score_two_agents(agent1, agent2, n_games = 10):
    env = h_env.HockeyEnv()

    score_history = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0

        while not done:
            agent1_action = agent1.choose_action(observation)

            agent2_action = agent2.choose_action(env.obs_agent_two())

            combined_action = np.hstack([agent1_action, agent2_action])

            observation_, reward, done, truncated, info = env.step(combined_action)
            done = done or truncated  # Check if the game is over
            score += reward

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history)
        print(f'Game {i + 1}, score {score:.2f}, avg score {avg_score:.2f}')

    env.close()

    return avg_score


def get_score_vs_bot(agent1, weak_opponent= False, n_games = 10):
    # Create the environment
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=weak_opponent)

    score_history = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        
        while not done:
            agent1_action = agent1.choose_action(observation)
            

            observation_, reward, done, truncated, info = env.step(agent1_action)
            done = done or truncated  # Check if the game is over
            score += reward

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history)
        print(f'Game {i + 1}, score {score:.2f}, avg score {avg_score:.2f}')

    env.close()

    return avg_score


def rank_agents(weight_directory, versions):

    env = h_env.HockeyEnv()
    #env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)


    # Initialize the agent
    agent1 = Agent(
        alpha=0,#config['alpha'],
        beta=config['beta'],
        input_dims=env.observation_space.shape,
        env=env,
        gamma=config['gamma'],
        n_actions=config['n_actions'],
        max_size=config['max_size'],
        tau=config['tau'],
        layer1_size=config['layer1_size'],
        layer2_size=config['layer2_size'],
        batch_size=config['batch_size'],
        reward_scale=config['reward_scale'],
        checkpoint_dir=config['checkpoint_dir']
    )

    agent2 = Agent(
        alpha=0,#config['alpha'],
        beta=config['beta'],
        input_dims=env.observation_space.shape,
        env=env,
        gamma=config['gamma'],
        n_actions=config['n_actions'],
        max_size=config['max_size'],
        tau=config['tau'],
        layer1_size=config['layer1_size'],
        layer2_size=config['layer2_size'],
        batch_size=config['batch_size'],
        reward_scale=config['reward_scale'],
        checkpoint_dir=config['checkpoint_dir']
    )
    score_dict = {}
    score_dict_vs_bot = {}
    for i in versions:#TODO: this does not have to be quadratic 
        scores = []
        print(f'Model {i} vs bot')
        agent1.load_models(file_path_actor=os.path.join(weight_directory, f'actor_sac_{i}'),
                                    file_path_critic1=os.path.join(weight_directory, f'critic_1_{i}'),
                                    file_path_critic2=os.path.join(weight_directory, f'critic_2_{i}'))
        
        score_dict_vs_bot[str(i)] = get_score_vs_bot(agent1, n_games = 20)
        for j in versions:
            print(f'Model {i} vs model {j}')
            agent2.load_models(file_path_actor=os.path.join(weight_directory, f'actor_sac_{j}'),
                                        file_path_critic1=os.path.join(weight_directory, f'critic_1_{j}'),
                                        file_path_critic2=os.path.join(weight_directory, f'critic_2_{j}'))

            
            score = get_score_two_agents(agent1, agent2, n_games = 20)
            scores.append(score)
        
        avg_score = np.mean(scores)

        score_dict[str(i)] = avg_score

    return score_dict, score_dict_vs_bot
        



if __name__ == '__main__':
    weight_directory = 'SAC/SelfMade/tmp/checkpoints'
    versions = [2600, 5000, 7400, 10000, 12400, 15000, 16400, 19800, 22000, 23000, 21000]    
    scores_vs_agents, scores_vs_bot = rank_agents(weight_directory, versions)
    print(dict(sorted(scores_vs_agents.items(), key=lambda item: item[1])))
    print(dict(sorted(scores_vs_bot.items(), key=lambda item: item[1])))