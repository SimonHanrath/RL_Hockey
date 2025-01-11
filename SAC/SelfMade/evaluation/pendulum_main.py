import gym
import numpy as np
from SAC.SelfMade.agent.agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Use the gym Pendulum environment
    env = gym.make('Pendulum-v1')#, render_mode='human')

    # Change this flag to True if you want to load the model and watch it play,
    # otherwise it will train from scratch.
    load_checkpoint = False

    # Initialize the agent
    agent = Agent(input_dims=env.observation_space.shape, env=env, 
                   n_actions=env.action_space.shape[0])
    n_games = 250
    filename = 'pendulum.png'
    figure_file = 'plots/' + filename

    best_score = -np.inf  # Pendulum rewards are typically negative
    score_history = []

    if load_checkpoint:
        agent.load_models()

        # Reset the environment before rendering
        observation = env.reset()[0]
        env.render()

        for i in range(n_games):
            observation = env.reset()[0]
            done = False
            score = 0
            while not done:
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
        # Train the agent from scratch and save checkpoints
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
            avg_score = np.mean(score_history[-50:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            print(f'episode {i}, score {score:.2f}, avg score {avg_score:.2f}')

        # Plotting
        plt.plot(score_history, marker='o', linestyle='-', color='b', label='Score History')
        plt.title('Score History')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()
