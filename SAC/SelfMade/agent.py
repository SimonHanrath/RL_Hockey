import os
import torch as T
import torch.nn.functional as F
import numpy as np
from replayBuffer import ReplayBuffer
from nets import ActorNet, CriticNet
import pickle

class Agent:
    """
    A reinforcement learning agent implementing the Soft Actor-Critic (SAC) algorithm.
    This agent includes the actor and critic networks, a replay buffer, and methods
    for training and saving/loading models.
    """

    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[18],
             env=None, gamma=0.99, n_actions=4, max_size=1000000, tau=0.005,
             layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, checkpoint_dir='SAC/SelfMade/tmp/checkpoints'):
        """
        Initializes the SAC agent with its networks, replay buffer, and hyperparameters.

        Args:
            alpha (float): Learning rate for the actor network.
            beta (float): Learning rate for the critic networks.
            input_dims (list): Dimensions of the input (state) space.
            env (gym.Env): Environment to derive action space from.
            gamma (float): Discount factor for future rewards.
            n_actions (int): Number of actions in the action space.
            max_size (int): Maximum size of the replay buffer.
            tau (float): Target network update rate.
            layer1_size (int): Number of neurons in the first hidden layer of networks.
            layer2_size (int): Number of neurons in the second hidden layer of networks.
            batch_size (int): Size of the training batch.
            reward_scale (float): Scale for rewards.
            checkpoint_dir (str): Directory to save and load model checkpoints.
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
        self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)  # Target entropy for SAC reduced for more exploration focus
        self.log_alpha = T.tensor(0.0, dtype=T.float32, requires_grad=True, device=self.actor.device)  # Log of alpha
        self.alpha = self.log_alpha.exp()  # Alpha, controlling exploration-exploitation tradeoff
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha)  # Optimizer for alpha


    def choose_action(self, observation):
        """
        Chooses an action based on the current policy (actor network).

        Args:
            observation (np.ndarray): Current state observation.

        Returns:
            np.ndarray: Selected action.
        """
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)  
        return actions.cpu().detach().numpy()[0] 

    def store(self, state, action, reward, new_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            new_state (np.ndarray): Next state.
            done (bool): Whether the episode has terminated.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, file_path_actor=None, file_path_critic1=None, file_path_critic2=None):
        """
        Saves the parameters of the actor and critic networks, replay buffer, and optimizer states to checkpoint files.
        """
        print('Saving models and optimizer states...')
        self.actor.save_checkpoint(file_path_actor)
        self.critic_1.save_checkpoint(file_path_critic1)
        self.critic_2.save_checkpoint(file_path_critic2)

    def load_models(self, file_path_actor=None, file_path_critic1=None, file_path_critic2=None):
        """
        Loads the parameters of the actor and critic networks from checkpoint files.
        """
        print('loading models ..')
        self.actor.load_checkpoint(file_path_actor)
        self.critic_1.load_checkpoint(file_path_critic1)
        self.critic_2.load_checkpoint(file_path_critic2)


    def clone(self):
        """
        Creates a clone of the current agent with identical weights and parameters.

        Returns:
            Agent: A new instance of the Agent class with the same weights and parameters.
        """
        # Create a new instance of the agent
        clone_agent = Agent(
            alpha=self.actor.optimizer.param_groups[0]['lr'],  # Learning rate for actor
            beta=self.critic_1.optimizer.param_groups[0]['lr'],  # Learning rate for critics
            input_dims=self.actor.input_dims,  # Input dimensions
            env=self.env,  # Use the environment from the current agent
            gamma=self.gamma,  # Discount factor
            n_actions=self.n_actions,  # Number of actions
            max_size=self.memory.mem_size,  # Replay buffer size
            tau=self.tau,  # Target network update rate
            layer1_size=self.actor.fc1_dims,  # First hidden layer size
            layer2_size=self.actor.fc2_dims,  # Second hidden layer size
            batch_size=self.batch_size,  # Batch size
            reward_scale=self.scale,  # Reward scaling factor
            checkpoint_dir=self.actor.checkpoint_dir  # Checkpoint directory
        )

        # Copy weights from the current agent to the cloned agent
        clone_agent.actor.load_state_dict(self.actor.state_dict())
        clone_agent.critic_1.load_state_dict(self.critic_1.state_dict())
        clone_agent.critic_2.load_state_dict(self.critic_2.state_dict())

        # Copy alpha and log_alpha values
        clone_agent.alpha = self.alpha.clone()
        clone_agent.log_alpha = self.log_alpha.clone()

        return clone_agent

        
    def learn(self, writer=None, step=None):
        """
        Updates the networks (actor, critics, and alpha) based on sampled experiences.

        Args:
            writer (SummaryWriter, optional): TensorBoard writer for logging.
            step (int, optional): Current training step for logging.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Compute target Q-values without a value network
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

        # Update temperature alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Logging
        if writer is not None and step is not None:
            writer.add_scalar('Loss/Actor_Loss', actor_loss.item(), step)
            writer.add_scalar('Loss/Alpha_Loss', alpha_loss.item(), step)
            writer.add_scalar('Temperature/Alpha', self.alpha.item(), step)
            writer.add_scalar('Loss/Critic_1_Loss', critic_1_loss.item(), step)
            writer.add_scalar('Loss/Critic_2_Loss', critic_2_loss.item(), step)