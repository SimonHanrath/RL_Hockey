import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import time

if __name__ == '__main__':
    # Choose your environment and mode
    # Example environment:
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)

    # Flags to control behavior
    test_mode = False        # If True, just run the model for testing (no training)
    resume_training = False  # If True and test_mode=False, load existing model and continue training

    # Initialize the agent
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=4)
    
    # Number of games to run
    n_games = 100
    best_score = -np.inf
    score_history = []
    avg_score_history = []

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/hockey_sac_training')

    # If testing or resuming training, load the model
    if test_mode or resume_training:
        print("Loading existing model weights...")
        agent.load_models()
        # If you have replay buffers or other states saved, load them here as well:
        # agent.load_replay_buffer("replay_buffer_path")
        # agent.load_optimizer_states("optimizer_states_path")
        
    if test_mode:
        # TEST MODE: Just run the agent, no training
        for i in range(n_games):
            observation = env.reset()[0]
            done = False
            score = 0
            env.render()
            while not done:
                time.sleep(0.1)
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
            print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')
        env.close()

    else:
        # TRAINING MODE: Either fresh start or resuming from loaded model
        for i in range(n_games):
            observation = env.reset()[0]
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                done = done or truncated
                score += reward

                # Store transition and learn
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()

                observation = observation_

            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
            avg_score_history.append(avg_score)

            # TensorBoard logging
            writer.add_scalar('Rewards/Episode_Score', score, i)
            writer.add_scalar('Rewards/Avg_Score', avg_score, i)

            # Save model if best score
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
                # Save any other training states if needed
                # agent.save_replay_buffer("replay_buffer_path")
                # agent.save_optimizer_states("optimizer_states_path")

            print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')

        writer.close()
        env.close()
        print("Training completed. Logs saved to TensorBoard.")
