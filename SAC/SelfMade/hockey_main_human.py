import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
import time

if __name__ == '__main__':
    # Create the environment
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL, keep_mode=True)

    # Initialize the agent
    agent = Agent(input_dims=env.observation_space.shape, env=env, 
                  n_actions=env.action_space.shape[0])

    # Load the agent's trained model
    agent.load_models()  # Ensure your agent's model is correctly trained or loaded here

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
            time.sleep(0.1)  # Slows the game for better human understanding
            env.render()

            # Human action for Player 1
            human_action = human_opponent.act(observation)

            # Agent action for Player 2
            agent_action = agent.choose_action(env.obs_agent_two())

            # Combine actions: Human controls Player 1, agent controls Player 2
            combined_action = np.hstack([human_action, agent_action])

            # Step the environment
            observation_, reward, done, truncated, info = env.step(combined_action)
            done = done or truncated  # Check if the game is over
            score += reward

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        print(f'Game {i + 1}, score {score:.2f}, avg score {avg_score:.2f}')

    env.close()
