import gym
import numpy as np
from SAC.SelfMade.agent.agent import Agent
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import random
import csv

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


def train_agent_self_play_league(agent, env, n_games=20000,
                          log_dir='runs/hockey_sac_training',
                          opponent_update_interval=20, log_file="training_log.csv"):
    """
    Train the SAC agent against a predifined set of opponents

    Args:
        agent:  SAC agent instance.
        env:   A HockeyEnv_BasicOpponent or HockeyEnv environment for self-play.
        n_games (int): Total number of training episodes.
        log_dir (str): Directory path for TensorBoard logging.
        opponent_update_interval (int): How often (in episodes) to update the older agent.
        """
    writer = SummaryWriter(log_dir)
    # Initialize log file for plotting later
    with open(log_file, mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["episode", "reward", 'opponent'])

    score_history = []

    # This is your built-in environment with a "hard" scripted bot:
    
    # Initial clone of agent -> older version
    opponent = agent.clone()
    wins = 0

    #TODO: do this somewhere else
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

    tmp = "model_weights/old_tests/SAC_selfplay2/checkpoints"
    agent2.load_models(file_path_actor=os.path.join(tmp, f'actor_sac_{15000}'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_{15000}'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_{15000}'))
    
    agent3 = Agent(
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

    tmp = "model_weights/old_tests/SAC_selfplay3/checkpoints"
    agent3.load_models(file_path_actor=os.path.join(tmp, f'actor_sac_{19800}'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_{19800}'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_{19800}'))
    

    agent4 = Agent(
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

    tmp = "model_weights/old_tests/SAC_selfplay_league1/checkpoints"
    agent4.load_models(file_path_actor=os.path.join(tmp, f'actor_sac_{19800}'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_{19800}'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_{19800}'))
    
    agent_trained_vs_strong_bot = Agent(
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

    tmp = "model_weights/for_report/standart/hard_bot"
    agent_trained_vs_strong_bot.load_models(file_path_actor=os.path.join(tmp, f'actor_sac'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_sac'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_sac'))
    

    agent_trained_vs_attack_bot = Agent(
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

    tmp = "model_weights/for_report/standart/vs_attack_bot"
    agent_trained_vs_attack_bot.load_models(file_path_actor=os.path.join(tmp, f'actor_sac'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_sac'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_sac'))
    

    agent_trained_vs_defense_bot = Agent(
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

    tmp = "model_weights/for_report/standart/vs_defend_bot"
    agent_trained_vs_defense_bot.load_models(file_path_actor=os.path.join(tmp, f'actor_sac'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_sac'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_sac'))
    

    agent_trained_vs_league1 = Agent(
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

    tmp = "model_weights/for_report/League/league_run_2/checkpoints"
    agent_trained_vs_league1.load_models(file_path_actor=os.path.join(tmp, f'actor_sac_19800'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_19800'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_19800'))
    
    agent_trained_vs_league2 = Agent(
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

    tmp = "model_weights/for_report/League/league_run_3/checkpoints"
    agent_trained_vs_league2.load_models(file_path_actor=os.path.join(tmp, f'actor_sac_23000'),
                                        file_path_critic1=os.path.join(tmp, f'critic_1_23000'),
                                        file_path_critic2=os.path.join(tmp, f'critic_2_23000'))
    
    
    

    
    
    



    game_memory_size = 20
    league = {'strong_bot' : {'env' : h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False), 'games':game_memory_size*[0], 'total_games':0, 'self': None},
              'weak_bot' : {'env' : h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=True), 'games':game_memory_size*[0], 'total_games':0, 'self': None},
              'def_bot' : {'env' : h_env.HockeyEnv_BasicOpponent(mode=1, weak_opponent=False), 'games':game_memory_size*[0], 'total_games':0, 'self': None},
              'atk_bot' : {'env' : h_env.HockeyEnv_BasicOpponent(mode=2, weak_opponent=False), 'games':game_memory_size*[0], 'total_games':0, 'self': None},
              'prev_self' : {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': opponent},
              'agent2': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent2},
              'agent3': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent3},
              'agent4': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent4},
              'agent5_shit_agent': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': opponent},
              'agent_trained_vs_strong_bot': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent_trained_vs_strong_bot},
              'agent_trained_vs_attack_bot': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent_trained_vs_attack_bot},
              'agent_trained_vs_defense_bot': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent_trained_vs_defense_bot},
              'agent_trained_vs_league1': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent_trained_vs_league1},
              'agent_trained_vs_league2': {'env' : env, 'games':game_memory_size*[0], 'total_games':0, 'self': agent_trained_vs_league2}
              }
    
    for i in range(n_games):

        
        
        #opponent_weights = [1/(0.1+np.mean(opponent['games'])) for opponent in league.values()] # maybe make the 0.1 into hyperparameter?
        opponent_weights = [game_memory_size-np.sum(opponent['games'])+1 for opponent in league.values()] # more games won = less likely to play against that agent
        normalized_weights = [weight / sum(opponent_weights) for weight in opponent_weights]

        opponent = random.choices(list(league.keys()), weights=normalized_weights, k=1)[0]
        print(opponent)

        score = run_episode(
                agent=agent,
                env=league[opponent]['env'],
                opponent=league[opponent]['self'], 
                episode_index=i,
                writer=writer
            )

        
        #remove oldest element from list of games
        del league[opponent]['games'][0]
        
        #increase game counter
        league[opponent]['total_games']+=1
        
        # agent "wins" if score >= 0 (here we count overall win rate and specific win rate)
        if score >= 0:
            wins += 1
            league[opponent]['games'].append(1)
        else:
            league[opponent]['games'].append(0)

        
        with open(log_file, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([i,score, opponent])

        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar('Rewards/Episode_Score', score, i)
        writer.add_scalar('Rewards/Avg_Score', avg_score, i)
        
        #log win rates against the opponents
        for opponent in league.keys():
            writer.add_scalar(f"games_against/{opponent}", league[opponent]['total_games'], global_step=i)
            writer.add_scalar(f"win_rates_against/{opponent}", np.mean(league[opponent]['games']) , global_step=i)

        #

        # Opponent update & save models periodically
        if i % opponent_update_interval == 0 and i > 0:
            win_rate = wins / opponent_update_interval
            writer.add_scalar(f'win_rate_last_{opponent_update_interval}', win_rate, i)
            wins = 0
            print(f"[Episode {i}] Win rate over last {opponent_update_interval} = {win_rate:.2f}, "
                  f"Avg Score: {avg_score:.2f}")
            print(f"Updating opponent at Episode {i}...")
            league['prev_self']['self'] = agent.clone()

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
    print("Training completed. Logs saved to TensorBoard.")


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
    
    train_agent_self_play_league(agent, env, n_games=config['n_games'], log_dir=config['log_dir'],
                           opponent_update_interval=config['opponent_update_interval'])
    

    #train_agent(agent, env, n_games=config['n_games'], log_dir=config['log_dir'])


