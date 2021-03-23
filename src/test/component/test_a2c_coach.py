from bomberman_rl.simulator.a2c_single_coach import A2CSingleCoach
from bomberman_rl.agent.a2c_agent import A2CAgent

# Getting agent
agent = A2CAgent(5, 5)

# Running train
coach = A2CSingleCoach("bomberman_rl:bomberman-minimal-v0", agent, "stdout",
                                  show_each=500, nb_passes=50000, fps=10, gamma=0.99)
coach.run()
