import gym
import torch
from bomberman_rl.agent.policy_gradient_agent import PolicyGradientAgent


env = gym.make("bomberman_rl:bomberman-default-v0")

# Checks if the environment resets without error
obs = env.reset()

agent = PolicyGradientAgent(*obs[:, :, 0].shape)

print(agent.choose_action(obs))
