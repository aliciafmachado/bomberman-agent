import gym
from bomberman_rl.simulator.dql_agent_single_coach import DQLAgentSingleCoach
from bomberman_rl.agent.dqn_agent import DQNAgent


env = gym.make("bomberman_rl:bomberman-minimal-v0")
show_loss = True
agent = DQNAgent(env.size[0], env.size[1], lr=1e-3, gamma=0.9)
coach = DQLAgentSingleCoach(env, agent, n_episodes=10, plot_loss=show_loss)
coach.run()

agent.save('./', 'dql_agent')
if show_loss:
    coach.show_loss()
