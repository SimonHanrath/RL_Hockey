import os
import torch as T
import torch.nn.functional as F
import numpy as np
from SAC.SelfMade.agent.replayBuffer import ReplayBuffer
from SAC.SelfMade.agent.nets import ActorNet, CriticNet
import pickle

class Agent:
    """
    A reinforcement learning agent implementing the Soft Actor-Critic (SAC) algorithm.
    This agent includes the actor and critic networks, a replay buffer, and methods
    for training and saving/loading models.
    """
    
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[18], env=None, gamma=0.99, n_actions=4, max_size=1000000, 
                 tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, 
                 checkpoint_dir='SAC/SelfMade/tmp/checkpoints'):
        """
        Initializes the SAC agent with its networks, replay buffer, and hyperparameters.
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        # initialize actor and critic networks
        self.actor = ActorNet(alpha, input_dims, n_actions=n_actions,
                              fc1_dims=layer1_size, fc2_dims=layer2_size,
                              name='actor', max_action=[1] * n_actions, checkpoint_dir=checkpoint_dir)
        self.critic_1 = CriticNet(beta, input_dims, n_actions=n_actions,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size,
                                  name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_2 = CriticNet(beta, input_dims, n_actions=n_actions,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size,
                                  name='critic_2', checkpoint_dir=checkpoint_dir)
        
        # automatic entropy temperature (alpha) tuning
        self.alpha = 0.1 # Fixed entropy temperature
    
    def choose_action(self, observation):
        """Chooses an action based on the current policy (actor network)."""
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def store(self, state, action, reward, new_state, done):
        """Stores a transition in the replay buffer."""
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def save_models(self, file_path_actor=None, file_path_critic1=None, file_path_critic2=None):
        """Saves the parameters of the actor and critic networks."""
        print('Saving models and optimizer states...')
        self.actor.save_checkpoint(file_path_actor)
        self.critic_1.save_checkpoint(file_path_critic1)
        self.critic_2.save_checkpoint(file_path_critic2)
    
    def load_models(self, file_path_actor=None, file_path_critic1=None, file_path_critic2=None):
        """Loads the parameters of the actor and critic networks."""
        print('loading models ..')
        self.actor.load_checkpoint(file_path_actor)
        self.critic_1.load_checkpoint(file_path_critic1)
        self.critic_2.load_checkpoint(file_path_critic2)
    
    def learn(self, writer=None, step=None):
        """Updates the networks (actor, critics, and alpha) based on sampled experiences."""
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        
        # Compute target Q-values
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            q1_next = self.critic_1.forward(state_, next_actions)
            q2_next = self.critic_2.forward(state_, next_actions)
            q_next = T.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = self.scale * reward + self.gamma * (1 - done.float()) * q_next.view(-1)
        
        # Update critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_old = self.critic_1.forward(state, action).view(-1)
        q2_old = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old, q_target)
        critic_2_loss = 0.5 * F.mse_loss(q2_old, q_target)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # Update actor
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new = self.critic_1.forward(state, actions)
        q2_new = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new, q2_new).view(-1)
        actor_loss = (self.alpha * log_probs.view(-1) - critic_value).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        # Logging
        if writer is not None and step is not None:
            writer.add_scalar('Loss/Actor_Loss', actor_loss.item(), step)
            writer.add_scalar('Temperature/Alpha', self.alpha, step)
            writer.add_scalar('Loss/Critic_1_Loss', critic_1_loss.item(), step)
            writer.add_scalar('Loss/Critic_2_Loss', critic_2_loss.item(), step)
