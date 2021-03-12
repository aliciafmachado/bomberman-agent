import gym
import torch
from bomberman_rl.agent.models.policy_model import Policy


env = gym.make("bomberman_rl:bomberman-default-v0")

# Checks if the environment resets without error
obs = env.reset()

# print()

policy = Policy(*obs[:, :, 0].shape).double()
print(policy(torch.tensor(obs).flatten().double()))
