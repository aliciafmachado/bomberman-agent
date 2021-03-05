import argparse
from bomberman_rl.simulator.single_agent_simulator import SingleAgentSimulator
from bomberman_rl.agent.trainable_agent import TrainableAgent

# Parsing arguments
parser = argparse.ArgumentParser(description='Run an agent from a file in a given environment')
parser.add_argument("--agent-name", help="agent's name", required=True)
parser.add_argument('--agent-path', help="path to the agent", default="./")
parser.add_argument('--environment', help="name of teh environment to be used",
                    default="bomberman_rl:bomberman-small-v0")
args = parser.parse_args()

# Loading agent and running simulation
agent = TrainableAgent.load(args.agent_path, args.agent_name)
simulator = SingleAgentSimulator(args.environment, agent, "print")

simulator.run()
