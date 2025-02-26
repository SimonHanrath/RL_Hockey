"""
File containing different model types that were used for training
"""

import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
class ActorCritic_single_hidden(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic_single_hidden, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
class ActorCritic_relu(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic_relu, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory != None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
    

#continuous version might not work
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCriticContinuous, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(n_latent_var, action_dim)
        self.std_layer = nn.Linear(n_latent_var, action_dim)

        # Critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(self.device)
        # Forward pass through actor layers
        actor_out = self.actor_layer(state)
        mean = torch.tanh(self.mean_layer(actor_out))  # Apply tanh to the mean to restrict the range
        std = self.std_layer(actor_out).exp()  # Standard deviation is positive, so we use exp

        # Create the action distribution (Gaussian)
        dist = Normal(mean, std)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action).sum(-1))  # Log probability sum for multi-dimensional actions

        return action.cpu().numpy()

    def evaluate(self, state, action):
        # Forward pass through actor layers
        actor_out = self.actor_layer(state)
        mean = torch.tanh(self.mean_layer(actor_out))  # Apply tanh to the mean
        std = self.std_layer(actor_out).exp()  # Exponentiate to ensure positive std

        # Create the action distribution
        dist = Normal(mean, std)

        action_logprobs = dist.log_prob(action).sum(-1)  # Log prob sum for multi-dimensional actions
        dist_entropy = dist.entropy().sum(-1)  # Entropy sum for multi-dimensional actions

        # Value estimate
        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
