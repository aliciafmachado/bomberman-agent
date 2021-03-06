import argparse
import gym
from bomberman_rl.agent.dqn_agent import DQNAgent
from bomberman_rl.simulator.dqn_agent_single_coach import DQNAgentSingleCoach
from bomberman_rl.simulator.single_agent_simulator import SingleAgentSimulator
from bomberman_rl.agent.trainable_agent import TrainableAgent

# Parsing arguments
parser = argparse.ArgumentParser(description='Run an agent from a file in a given environment')
parser.add_argument("--agent-name", help="agent's name", required=True)
parser.add_argument('--agent-path', help="path to the agent", default="./")
parser.add_argument('--environment', help="name of teh environment to be used",
                    default="bomberman_rl:bomberman-default-v0")
args = parser.parse_args()

# Loading agent and running simulation
agent = TrainableAgent.load(args.agent_path, args.agent_name)
env = gym.make(args.environment)

if agent.__class__ == DQNAgent:
    coach = DQNAgentSingleCoach(env, agent, fps=3, max_steps=250)
    coach.simulate(1)
else:
    simulator = SingleAgentSimulator(env, agent, "human")
    simulator.run()
