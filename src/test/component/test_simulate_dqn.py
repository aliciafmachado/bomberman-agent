import gym
from bomberman_rl.simulator.dqn_agent_single_coach import DQNAgentSingleCoach
from bomberman_rl.agent.dqn_agent import DQNAgent


env = gym.make("bomberman_rl:bomberman-minimal-v0")
agent = DQNAgent.load('./', 'dql_agent')
coach = DQNAgentSingleCoach(env, agent, n_episodes=10000)
coach.simulate(1)

