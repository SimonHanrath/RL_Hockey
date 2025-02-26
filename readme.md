# Team HAT 🎩
## PPO and SAC agent for 2D Hockey env

This repository contains two reinforcement learning algorithms, Soft Actor-Critic (SAC) and Proximal
 Policy Optimization(PPO). We evaluate their performance on a 2D hockey environment, where two
agents control paddles to hit a puck into the opponent’s goal. The observation space is 18 dimensional
and includes positions, velocities, and angular movements of both players and the puck. The hockey
environment supports continuous and discrete action spaces, where agents can either output real-valued
forces for movement, rotation, and shooting or select from predefined discrete actions mapped to equiv-
alent movements.

