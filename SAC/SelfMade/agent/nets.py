import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np



"""
-> Disclaimer: The basic structure was partially inspired by: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC.
"""


class ActorNet(nn.Module):
    """
    Actor network for the Soft Actor-Critic algorithm.
    This network generates actions and their log probabilities for a given state.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        mu (nn.Linear): Output layer producing the mean of the action distribution.
        sigma (nn.Linear): Output layer producing the standard deviation of the action distribution.
        optimizer (torch.optim.Adam): Optimizer for the network.
        device (torch.device): Device where the network is stored.
    """

    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
                 fc2_dims=256, n_actions=2, name='actor', checkpoint_dir='SAC/SelfMade/tmp/checkpoints'):
        """
        Initializes the Actor network with the given hyperparameters.

        Args:
            alpha (float): Learning rate for the optimizer.
            input_dims (list): Dimensions of the input state.
            max_action (float or list): Maximum action value(s) for the environment.
            fc1_dims (int): Number of neurons in the first hidden layer.
            fc2_dims (int): Number of neurons in the second hidden layer.
            n_actions (int): Number of actions in the action space.
            name (str): Name of the network for saving/loading checkpoints.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6  # noise added during reparameterization

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            tuple: Mean and standard deviation of the action distribution.
        """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)  # for numerical stability
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Samples actions and their log probabilities from the policy.

        Args:
            state (torch.Tensor): Input state tensor.
            reparameterize (bool): Whether to reparameterize the sampling process.

        Returns:
            tuple: Sampled action and its log probability.
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # scale action and compute log probabilities
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self, file_path=None):
        """Saves the network's parameters to a checkpoint file."""
        if file_path != None:
            T.save(self.state_dict(), file_path)
        else: 
            T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self, file_path=None):
        """Loads the network's parameters from a checkpoint file."""
        
        if file_path != None:
            self.load_state_dict(T.load(file_path))
        else: 
            self.load_state_dict(T.load(self.checkpoint_file))


class CriticNet(nn.Module):
    """
    Critic network for the Soft Actor-Critic algorithm.
    This network estimates the Q-value of a given (state, action) pair.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        q (nn.Linear): Output layer producing the Q-value.
        optimizer (torch.optim.Adam): Optimizer for the network.
        device (torch.device): Device where the network is stored.
    """

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', checkpoint_dir='SAC/SelfMade/tmp/checkpoints'):
        """
        Initializes the Critic network with the given hyperparameters.

        Args:
            beta (float): Learning rate for the optimizer.
            input_dims (list): Dimensions of the input state.
            n_actions (int): Number of actions in the action space.
            fc1_dims (int): Number of neurons in the first hidden layer.
            fc2_dims (int): Number of neurons in the second hidden layer.
            name (str): Name of the network for saving/loading checkpoints.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Estimated Q-value.
        """
        x = self.fc1(T.cat([state, action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.q(x)

    def save_checkpoint(self, file_path=None):
        """Saves the network's parameters to a checkpoint file."""
        if file_path != None:
            T.save(self.state_dict(), file_path)
        else: 
            T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self, file_path=None):
        """Loads the network's parameters from a checkpoint file."""
        
        if file_path != None:
            self.load_state_dict(T.load(file_path))
        else: 
            self.load_state_dict(T.load(self.checkpoint_file))

