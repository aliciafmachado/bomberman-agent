import gym
from bomberman_rl.simulator.dqn_agent_single_coach import DQNAgentSingleCoach
from bomberman_rl.agent.dqn_agent import DQNAgent


env = gym.make("bomberman_rl:bomberman-minimal-v0")
show_loss = True
agent = DQNAgent(env.size[0], env.size[1], lr=1e-3, gamma=0.9)
coach = DQNAgentSingleCoach(env, agent, n_episodes=1000, plot_loss=show_loss)
coach.run()

agent.save('./', 'dql_agent')
if show_loss:
    coach.show_loss()
