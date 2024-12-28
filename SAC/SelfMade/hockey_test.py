import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
import time
import yaml

with open('SAC/SelfMade/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    # Create the environment
    env = h_env.HockeyEnv()

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

    # Load the agent's trained model
    agent1.load_models()  # Ensure your agent's model is correctly trained or loaded here
    agent2.load_models()

    # Initialize human as Player 1 (left side)
    human_opponent = h_env.HumanOpponent(env, player=1)  # Human controls Player 1

    n_games = 10  # Number of games to play
    score_history = []
    avg_score_history = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        print(f"\nStarting Game {i + 1}...")

        while not done:
            time.sleep(0.1)  # I am old and need this
            env.render()

            agent1_action = agent1.choose_action(observation)

            agent2_action = agent1.choose_action(env.obs_agent_two())

            human_action = human_opponent.act(observation)

            combined_action = np.hstack([agent1_action, agent2_action])
            #combined_action = np.hstack([human_action, agent2_action])

            observation_, reward, done, truncated, info = env.step(combined_action)
            done = done or truncated  # Check if the game is over
            score += reward

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        print(f'Game {i + 1}, score {score:.2f}, avg score {avg_score:.2f}')

    env.close()
