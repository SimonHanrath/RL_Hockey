import hockey.hockey_env as h_env
import multiprocessing

def large_discrete_to_continous_action(env, discrete_action):
    ''' converts discrete actions into continuous ones (for each player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.
        This is surely limiting. Other discrete actions are possible
        Action 0: do nothing
        Action 1: -1 in x
        Action 2: -1 in x and -1 in y
        Action 3: -1 in x and 1 in y
        Action 4: -1 in x and -1 in angle
        Action 5: -1 in x and 1 in angle
        Action 6: -1 in x and -1 in y and -1 in angle
        Action 7: -1 in x and -1 in y and 1 in angle
        Action 8: -1 in x and 1 in y and -1 in angle
        Action 9: -1 in x and 1 in y and 1 in angle
        Action 10: 1 in x
        Action 11: 1 in x and -1 in y
        Action 12: 1 in x and 1 in y
        Action 13: 1 in x and -1 in angle
        Action 14: 1 in x and 1 in angle
        Action 15: 1 in x and -1 in y and -1 in angle
        Action 16: 1 in x and -1 in y and 1 in angle
        Action 17: 1 in x and 1 in y and -1 in angle
        Action 18: 1 in x and 1 in y and 1 in angle
        Action 19: -1 in y
        Action 20: -1 in y and -1 in angle
        Action 21: -1 in y and 1 in angle
        Action 22: 1 in y
        Action 23: 1 in y and -1 in angle
        Action 24: 1 in y and 1 in angle
        Action 25: -1 in angle
        Aciton 26: 1 in angle
        Action 27: shoot (if keep_mode is on)

        
        '''
    actions_x_is_neg1 = range(1,10)
    actions_x_is_1 = range(10,19)
    actions_y_is_neg1 = [2,6,7,11,15,16,19,20,21]
    actions_y_is_1 = [3,5,9,12,17,18,22,23,24]
    actions_angle_is_neg1 = [4,6,8,13,15,17,20,23,25]
    actions_angle_is_1 = [5,7,9,14,16,18,21,24,26]

    if discrete_action in actions_x_is_neg1:
        x = -1.
    elif discrete_action in actions_x_is_1:
        x = 1.
    else:
        x = 0.

    if discrete_action in actions_y_is_neg1:
        y = -1.
    elif discrete_action in actions_y_is_1:
        y = 1.
    else:
        y = 0.

    if discrete_action in actions_angle_is_neg1:
        angle = -1.
    elif discrete_action in actions_angle_is_1:
        angle = 1.
    else:
        angle = 0.

    action_cont = [x,y,angle] 
    if env.keep_mode:
      action_cont.append((discrete_action == 27) * 1.0)

    return action_cont

def custom_reward(env, defense_mode, info, sparse_reward = False):
    reward_win = 0

    if env.done:
      if env.winner == 0:  # tie
        reward_win += 0
      elif env.winner == 1:  # you won
        reward_win += 20
      else:  # opponent won
        reward_win -= 20

    reward_closeness_to_goal = 0

    if True: # defense_mode and env.player1_has_puck <= 1:
        dist_to_goal = h_env.dist_positions(env.player1.position, env.goal_player_1.position)
        max_dist = 250. / h_env.SCALE
        max_reward = -10.  
        factor = max_reward / (max_dist * env.max_timesteps / 2)
        reward_closeness_to_goal += dist_to_goal * factor  

    reward_touch_puck = 0.
    if (env.player1_has_puck == h_env.MAX_TIME_KEEP_PUCK) and defense_mode:
      reward_touch_puck = 10.

    reward_puck_closeness = info["reward_closeness_to_puck"]

    if sparse_reward:
       total_reward = 5*(reward_win+reward_touch_puck)
    else:
       total_reward = reward_win + reward_closeness_to_goal + reward_touch_puck + reward_puck_closeness

    return total_reward
