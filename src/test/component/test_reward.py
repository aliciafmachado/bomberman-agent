import gym

from bomberman_rl.envs.conventions import RIGHT, DOWN, STOP, PLACE_BOMB, LEFT, UP
from bomberman_rl.agent.controllable_agent import ControllableAgent

env = gym.make("bomberman_rl:bomberman-default-v0", n_agents=2)
agent1 = ControllableAgent()
actions2 = STOP
done = False

observations = env.reset()
while not done:
    observations, rewards, done, _ = env.step([agent1.choose_action(),
                                               STOP])
    env.render(steps_per_sec=4)