"""
Copy of the tester script that is able to generate videos
"""
import sys
import os
sys.path.append(os.getcwd())

import argparse
import torch
import numpy as np
import sys
import os
import hockey.hockey_env as h_env
import model
import helper_functions as functions
import imageio.v2 as imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_policy(checkpoint_path, enemy_checkpoint_path, num_episodes, visualize, action_type, use_default_enemy, record_video):
    
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=False)  
    state_dim = env.observation_space.shape[0]

    if action_type == "large_discrete":
        action_dim = 28
    else:
        action_dim = 8  

    n_latent_var = 256

    if visualize:
        num_episodes = 20

    # Load policies
    policy1 = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)
    policy1.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy1.eval()

    policy2 = model.ActorCritic(state_dim, action_dim, n_latent_var).to(device)
    policy2.load_state_dict(torch.load(enemy_checkpoint_path, map_location=device))
    policy2.eval()

    match_history = [0,0,0]

    if record_video:
        video_filename = f"Gameplay Video.mp4"
        writer = imageio.get_writer(video_filename, fps=30)

    # Run games and record scores
    for episode in range(1, num_episodes + 1):
        state, _info = env.reset()
        episode_reward = 0

        if episode%100 == 0:
            print(str(episode) + "games finished")

        for t in range(500):

            # Render the environment if wanted with argument --visualize 1
            if visualize:
                env.render(mode="rgb_array")

            if record_video:
                frame = env.render(mode="rgb_array")
                writer.append_data(frame)

            # Get action from the policy
            discrete_action1 = policy1.act(state)
            discrete_action2 = policy2.act(env.obs_agent_two())
            if action_type == "large_discrete":
                continuous_action1 = functions.large_discrete_to_continous_action(env, discrete_action1)
                continuous_action2 = functions.large_discrete_to_continous_action(env, discrete_action2)
            else:
                continuous_action1 = env.discrete_to_continous_action(discrete_action1)
                continuous_action2 = env.discrete_to_continous_action(discrete_action2)

            # Step environment
            if not(use_default_enemy):
                combined_action = np.hstack([continuous_action1, continuous_action2])
            else:
                combined_action = continuous_action1
            next_state, reward, done, _trunc, _info = env.step(combined_action)

            state = next_state
            episode_reward += reward

            if done:
                if env.winner == 0: 
                    match_history[1] += 1
                elif env.winner == 1:  
                    match_history[0] += 1
                else: 
                    match_history[2] += 1
                break

    
    #print final evaluation
    winrate = match_history[0]/(match_history[0]+match_history[2])
    print("Match history" + str(match_history) + " with winrate " + str(winrate))
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="PPO/checkpoints/gamma_runs/gamma_0.96/best_PPO_hockey-eps0.2.pth")
    parser.add_argument('--enemy_checkpoint_path', type=str, default="PPO/checkpoints/gamma_runs/gamma_0.96/best_PPO_hockey-eps0.2.pth")
    parser.add_argument('--use_default_enemy', type=int, default=1, help='0 or 1')
    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--action_type', type=str, default="large_discrete")
    parser.add_argument('--record_video', type=int, default=1, help='0 or 1')

    args = parser.parse_args()


    num_episodes = args.num_episodes
    checkpoint_path = args.checkpoint_path
    enemy_checkpoint_path = args.enemy_checkpoint_path
    visualize = bool(args.visualize)
    action_type = args.action_type
    use_default_enemy = args.use_default_enemy
    record_video = bool(args.record_video)
    

    test_policy(checkpoint_path, enemy_checkpoint_path, num_episodes, visualize, action_type, use_default_enemy, record_video)
