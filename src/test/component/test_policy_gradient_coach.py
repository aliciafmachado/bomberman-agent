from bomberman_rl.simulator.policy_gradient_single_coach import PolicyGradientSingleCoach
from bomberman_rl.agent.policy_gradient_agent import PolicyGradientAgent

# Getting agent
agent = PolicyGradientAgent(5, 5)

# Running train
coach = PolicyGradientSingleCoach("bomberman_rl:bomberman-minimal-v0", agent, "stdout",
                                  show_each=500, nb_passes=50000, fps=10, gamma=0.99)
coach.run()
