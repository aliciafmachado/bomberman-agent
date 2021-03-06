import gym

from bomberman_rl.agent.controllable_agent import ControllableAgent


env = gym.make("bomberman_rl:bomberman-default-v0")
agent = ControllableAgent()
done = False

env.reset()
while not done:
    _, _, done, _ = env.step(agent.choose_action())
    env.render(steps_per_sec=4)
