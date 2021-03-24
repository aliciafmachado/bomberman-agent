import argparse
import gym
from bomberman_rl.simulator.q_agent_single_coach import QAgentSingleCoach
from bomberman_rl.agent.q_agent import QAgent

# Parsing arguments
parser = argparse.ArgumentParser(description='Trains a QAgent to destroy blocks alone')
parser.add_argument("--agent-name", help="agent file name after train", required=True)
parser.add_argument('--agent-path', help="path to agent file after train", default="./")
parser.add_argument("--agent-pretrained-name", help="name of pretrained agent to continue training", default=None)
parser.add_argument('--agent-pretrained-path', help="path of pretrained agent to continue training", default="./")
parser.add_argument('--environment', help="name of the environment to be used",
                    default="bomberman_rl:bomberman-default-v0")
parser.add_argument("--display", help="'human' or 'stdout'", default="human")
args = parser.parse_args()

# Loading agent if needed or creating new one
agent = None
if args.agent_pretrained_name:
    agent = QAgent.load(args.agent_pretrained_path, args.agent_pretrained_name)
else:
    agent = QAgent()

# Running train
env = gym.make(args.environment)
coach = QAgentSingleCoach(env, agent, args.display, nb_passes=5000)
coach.run()

# Saving result
agent.save(args.agent_path, args.agent_name)
