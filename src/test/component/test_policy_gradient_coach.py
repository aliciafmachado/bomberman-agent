import gym
from bomberman_rl.simulator.policy_gradient_single_coach import PolicyGradientSingleCoach
from bomberman_rl.agent.policy_gradient_agent import PolicyGradientAgent

# Getting agent
agent = PolicyGradientAgent(11, 13)

# Running train
env = gym.make("bomberman_rl:bomberman-default-v0")
coach = PolicyGradientSingleCoach(env, agent, "stdout",
                                  show_each=500, nb_passes=5000, fps=10, gamma=0.99)
coach.run()
