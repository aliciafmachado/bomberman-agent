import gym

from bomberman_rl.agent.controllable_agent import ControllableAgent
from bomberman_rl.agent.q_agent import QAgent


env = gym.make("bomberman_rl:bomberman-minimal-v0", n_agents=2)
agent1 = ControllableAgent()
agent2 = QAgent()
agent2.switch_mode('eval')
done = False

observations = env.reset()
while not done:
    observations, rewards, done, _ = env.step([agent1.choose_action(),
                                               agent2.choose_action(observations[1])])
    env.render(steps_per_sec=4)
