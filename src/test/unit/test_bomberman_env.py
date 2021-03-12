import gym

env = gym.make("bomberman_rl:bomberman-default-v0")

# Checks if the environment resets without error
env.reset()
