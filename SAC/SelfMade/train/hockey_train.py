import numpy as np
from SAC.SelfMade.agent.agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import csv


"""
python -m SAC.SelfMade.train.hockey_train
"""

# load config file for model and training specifications
with open(os.path.join('SAC', 'SelfMade', 'train', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)






def run_episode(agent, env, opponent=None, episode_index=0, writer=None):
    """
    Runs one episode in the given environment with the specified agent.
    If 'opponent' is provided, uses self-play with that opponent.
    Otherwise, plays against the environment's built-in AI.
    
    Returns:
        (float) final score of the episode
    """
    obs = env.reset()[0]
    done = False
    truncated = False
    score = 0.0

    while not done:
        agent_action = agent.choose_action(obs)

        if opponent is not None:  # self-play with older version
            opp_action = opponent.choose_action(env.obs_agent_two())
            combined_action = np.hstack([agent_action, opp_action])
            obs_, reward, done, truncated, info = env.step(combined_action)
        else: # Agent vs. built-in bot
            obs_, reward, done, truncated, info = env.step(agent_action)

        done = done or truncated
        score += reward

        
        agent.store(obs, agent_action, reward, obs_, done)
        agent.learn(writer=writer, step=episode_index)

        #env.render()
        obs = obs_

    return score


def train_agent_self_play(agent, env, n_games=20000,
                          log_dir='runs/hockey_sac_training',
                          opponent_update_interval=20, log_file="training_log.csv"):
    """
    Train the SAC agent, alternating between self-play (agent vs. older agent)
    and playing against a built-in 'hard mode' bot environment.

    Args:
        agent:  SAC agent instance.
        env:   A HockeyEnv_BasicOpponent or HockeyEnv environment for self-play.
        n_games (int): Total number of training episodes.
        log_dir (str): Directory path for TensorBoard logging.
        opponent_update_interval (int): How often (in episodes) to update the older agent.
        """
    writer = SummaryWriter(log_dir)
    score_history = []

    # This is your built-in environment with a "hard" scripted bot:
    env_bot = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)

    # Initial clone of agent -> older version
    opponent = agent.clone()
    wins = 0

    for i in range(n_games):
        # Randomly pick whether to do self-play or agent-vs-bot
        is_self_play = (np.random.randint(0, 2) == 0)

        if is_self_play:
            score = run_episode(
                agent=agent,
                env=env,
                opponent=opponent, 
                episode_index=i,
                writer=writer
            )
        else:
            print('playing against bot')
            score = run_episode(
                agent=agent,
                env=env_bot,
                opponent=None,
                episode_index=i,
                writer=writer
            )

        # agent "wins" if score >= 0
        if score >= 0:
            wins += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar('Rewards/Episode_Score', score, i)
        writer.add_scalar('Rewards/Avg_Score', avg_score, i)

        # Opponent update & save models periodically
        if i % opponent_update_interval == 0 and i > 0:
            win_rate = wins / opponent_update_interval
            writer.add_scalar(f'win_rate_last_{opponent_update_interval}', win_rate, i)
            wins = 0
            print(f"[Episode {i}] Win rate over last {opponent_update_interval} = {win_rate:.2f}, "
                  f"Avg Score: {avg_score:.2f}")
            print(f"Updating opponent at Episode {i}...")
            opponent = agent.clone()

            if i % 200 == 0:
                print(f"[Episode {i}] Saving model checkpoints...")
                agent.save_models(
                    file_path_actor=os.path.join(agent.checkpoint_dir, f'actor_sac_{i}'),
                    file_path_critic1=os.path.join(agent.checkpoint_dir, f'critic_1_{i}'),
                    file_path_critic2=os.path.join(agent.checkpoint_dir, f'critic_2_{i}')
                )

        print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

    writer.close()
    env.close()
    env_bot.close()
    print("Training completed. Logs saved to TensorBoard.")



def train_agent(agent, env, n_games=20000, log_dir='runs/hockey_sac_training', log_file="training_log.csv"):
    """
    Train the SAC agent by playing against whatever default behavior 'env' offers.
    In many cases, this might be a single-agent environment or a built-in bot.
    """
    writer = SummaryWriter(log_dir)
    best_score = -np.inf
    score_history = []

    # Initialize log file for plotting later
    with open(log_file, mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["epsiode", "reward"])

    for i in range(n_games):
        score = run_episode(
            agent=agent,
            env=env,
            opponent=None,
            episode_index=i,
            writer=writer
        )

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        writer.add_scalar('Rewards/Episode_Score', score, i)
        writer.add_scalar('Rewards/Avg_Score', avg_score, i)

        # save models if we have a new "best" average score
        if avg_score > best_score + 0.5:
            best_score = avg_score
            print("[train_agent] New best average score. Saving model...")
            agent.save_models()

        print(f"[train_agent] Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

        # add reward to csv file
        with open(log_file, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([i,score])

    writer.close()
    env.close()
    print("[train_agent] Training completed. Logs saved to TensorBoard.")


if __name__ == '__main__':

    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=True)
    
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

    
    #train_agent_self_play(agent, env, n_games=config['n_games'], log_dir=config['log_dir'],
    #                       opponent_update_interval=config['opponent_update_interval'])
    

    train_agent(agent, env, n_games=config['n_games'], log_dir=config['log_dir'])



