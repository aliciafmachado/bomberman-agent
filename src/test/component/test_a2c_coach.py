import gym
from bomberman_rl.simulator.a2c_single_coach import A2CSingleCoach
from bomberman_rl.agent.a2c_agent import A2CAgent

# Getting agent
agent = A2CAgent(11, 13)

# Running train
env = gym.make("bomberman_rl:bomberman-default-v0")
coach = A2CSingleCoach(env, agent, "stdout",
                       show_each=500, nb_passes=5000, fps=10, gamma=0.99)
coach.run()

