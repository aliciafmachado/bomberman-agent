import gym
from bomberman_rl.agent.a2c_agent import A2CAgent


env = gym.make("bomberman_rl:bomberman-default-v0")

# Checks if the environment resets without error
obs = env.reset()

agent = A2CAgent(*obs[:, :, 0].shape)

print(agent.choose_action(obs))
