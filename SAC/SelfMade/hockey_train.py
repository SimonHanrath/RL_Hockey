import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import yaml
import time

with open('SAC/SelfMade/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def train_agent_self_play(agent, env, n_games=20000, resume_training=False, log_dir='runs/hockey_sac_training', opponent_update_interval=20, win_rate_threshold=0.9):
    """Train the SAC agent with self-play."""
    writer = SummaryWriter(log_dir)
    best_score = -np.inf
    score_history = []
    avg_score_history = []

    # Resume training if specified
    if resume_training:
        print("Loading existing model weights...")
        agent.load_models()

    # Clone the agent to create the opponent
    opponent = agent.clone()
    wins = 0  # Track wins for dynamic updates

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        games = 0

        while not done:
            # Main agent (agent1) and opponent (agent2) actions
            agent1_action = agent.choose_action(observation)
            agent2_action = opponent.choose_action(env.obs_agent_two())

            # Combine actions and step the environment
            combined_action = np.hstack([agent1_action, agent2_action])
            observation_, reward, done, truncated, info = env.step(combined_action)
            done = done or truncated
            score += reward

            # Store transitions for the main agent
            agent.store(observation, agent1_action, reward, observation_, done)
            agent.learn(writer=writer, step=i)

            env.render()

            # Update observation
            observation = observation_

        # Track win rate
        if score >= 0:  # Assuming positive score means the main agent won
            wins += 1
        games += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        writer.add_scalar('Rewards/Episode_Score', score, i)
        writer.add_scalar('Rewards/Avg_Score', avg_score, i)

        # Update opponent if necessary
        if i % opponent_update_interval == 0:
            win_rate = wins / opponent_update_interval
            writer.add_scalar(f'win_rate_last_{opponent_update_interval}', win_rate, i)
            wins = 0
            print(f"Win rate: {win_rate:.2f}, Avg Score: {avg_score:.2f}")    
            if win_rate >= win_rate_threshold:
                print(f"Updating opponent at Episode {i}...")
                opponent = agent.clone()
                print(f'[Episode {i}] Saving best model...')
                agent.save_models()

        print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')

    writer.close()
    env.close()
    print("Training completed. Logs saved to TensorBoard.")





def train_agent(agent, env, n_games=20000, resume_training=False, log_dir='runs/hockey_sac_training'):
    """Train the SAC agent."""
    writer = SummaryWriter(log_dir)
    best_score = -np.inf
    score_history = []
    avg_score_history = []

    if resume_training: # TODO: currently does not load replay buffer and optimizer state which is a problem
        print("Loading existing model weights...")
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            done = done or truncated
            score += reward

            agent.store(observation, action, reward, observation_, done)
            agent.learn(writer=writer, step=i)

            #env.render()

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        writer.add_scalar('Rewards/Episode_Score', score, i)
        writer.add_scalar('Rewards/Avg_Score', avg_score, i)

        if avg_score > best_score + 0.5:
            best_score = avg_score
            print('Saving best model...')
            agent.save_models()

        print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')

    writer.close()
    env.close()
    print("Training completed. Logs saved to TensorBoard.")

if __name__ == '__main__':

    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)
    
    agent = Agent(
        alpha=config['alpha'],
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
    
    train_agent_self_play(agent, env, n_games=config['n_games'], resume_training=config['resume_training'],
                           log_dir=config['log_dir'], opponent_update_interval=config['opponent_update_interval'], win_rate_threshold=config['win_rate_threshold'])
    

    #train_agent(agent, env, n_games=config['n_games'], resume_training=config['resume_training'],
    #                       log_dir=config['log_dir'])


