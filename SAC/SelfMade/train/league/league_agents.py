import os
import numpy as np
import hockey.hockey_env as h_env
from SAC.SelfMade.agent.agent import Agent

def get_league(env, agent, config):
    """
    Builds and returns a dictionary of league opponents.
    
    Args:
        env: The environment instance.
        agent: The main SAC agent (used for cloning its previous self).
        config: Dictionary with configuration parameters.
    
    Returns:
        A dictionary representing the league.
    """
    # Clone the current agent as the "previous self" opponent.
    opponent = agent.clone()

    def load_agent(checkpoint_dir, actor_name, critic1_name, critic2_name):
        new_agent = Agent(
            alpha=0,  # using 0 as in your original code
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
        new_agent.load_models(
            file_path_actor=os.path.join(checkpoint_dir, actor_name),
            file_path_critic1=os.path.join(checkpoint_dir, critic1_name),
            file_path_critic2=os.path.join(checkpoint_dir, critic2_name)
        )
        return new_agent

    # Load agents from saved checkpoints.
    agent2 = load_agent("model_weights/old_tests/SAC_selfplay2/checkpoints",
                        f'actor_sac_{15000}', f'critic_1_{15000}', f'critic_2_{15000}')
    
    agent3 = load_agent("model_weights/old_tests/SAC_selfplay3/checkpoints",
                        f'actor_sac_{19800}', f'critic_1_{19800}', f'critic_2_{19800}')
    
    agent4 = load_agent("model_weights/old_tests/SAC_selfplay_league1/checkpoints",
                        f'actor_sac_{19800}', f'critic_1_{19800}', f'critic_2_{19800}')
    
    agent_trained_vs_strong_bot = load_agent("model_weights/for_report/standart/hard_bot",
                        f'actor_sac', f'critic_1_sac', f'critic_2_sac')
    
    agent_trained_vs_attack_bot = load_agent("model_weights/for_report/standart/vs_attack_bot",
                        f'actor_sac', f'critic_1_sac', f'critic_2_sac')
    
    agent_trained_vs_defense_bot = load_agent("model_weights/for_report/standart/vs_defend_bot",
                        f'actor_sac', f'critic_1_sac', f'critic_2_sac')
    
    agent_trained_vs_league1 = load_agent("model_weights/for_report/League/league_run_2/checkpoints",
                        f'actor_sac_19800', f'critic_1_19800', f'critic_2_19800')
    
    agent_trained_vs_league2 = load_agent("model_weights/for_report/League/league_run_3/checkpoints",
                        f'actor_sac_23000', f'critic_1_23000', f'critic_2_23000')

    game_memory_size = 20
    league = {
        'strong_bot': {'env': h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False),
                       'games': game_memory_size * [0],
                       'total_games': 0,
                       'self': None},
        'weak_bot': {'env': h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=True),
                     'games': game_memory_size * [0],
                     'total_games': 0,
                     'self': None},
        'def_bot': {'env': h_env.HockeyEnv_BasicOpponent(mode=1, weak_opponent=False),
                    'games': game_memory_size * [0],
                    'total_games': 0,
                    'self': None},
        'atk_bot': {'env': h_env.HockeyEnv_BasicOpponent(mode=2, weak_opponent=False),
                    'games': game_memory_size * [0],
                    'total_games': 0,
                    'self': None},
        'prev_self': {'env': env,
                      'games': game_memory_size * [0],
                      'total_games': 0,
                      'self': opponent},
        'agent2': {'env': env,
                   'games': game_memory_size * [0],
                   'total_games': 0,
                   'self': agent2},
        'agent3': {'env': env,
                   'games': game_memory_size * [0],
                   'total_games': 0,
                   'self': agent3},
        'agent4': {'env': env,
                   'games': game_memory_size * [0],
                   'total_games': 0,
                   'self': agent4},
        'agent5_shit_agent': {'env': env,
                              'games': game_memory_size * [0],
                              'total_games': 0,
                              'self': opponent},
        'agent_trained_vs_strong_bot': {'env': env,
                                        'games': game_memory_size * [0],
                                        'total_games': 0,
                                        'self': agent_trained_vs_strong_bot},
        'agent_trained_vs_attack_bot': {'env': env,
                                        'games': game_memory_size * [0],
                                        'total_games': 0,
                                        'self': agent_trained_vs_attack_bot},
        'agent_trained_vs_defense_bot': {'env': env,
                                         'games': game_memory_size * [0],
                                         'total_games': 0,
                                         'self': agent_trained_vs_defense_bot},
        'agent_trained_vs_league1': {'env': env,
                                     'games': game_memory_size * [0],
                                     'total_games': 0,
                                     'self': agent_trained_vs_league1},
        'agent_trained_vs_league2': {'env': env,
                                     'games': game_memory_size * [0],
                                     'total_games': 0,
                                     'self': agent_trained_vs_league2}
    }
    
    return league
