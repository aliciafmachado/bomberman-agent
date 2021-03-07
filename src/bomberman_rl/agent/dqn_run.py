# Run main for DQNAgent

from bomberman_rl.memory.ReplayMemory import ReplayMemory
from bomberman_rl.memory.ReplayMemory import Simulation
from bomberman_rl.agent.DQNAgent import DQNAgent
import torchvision.transforms as transforms
import gym
import argparse
import torch
import os

def main():
    # Parser
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("nb_episodes", type=int)

    # Optional arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--memory_size", type=int, default=10000)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)


    args = parser.parse_args()

    # Args
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create a transform to convert the matrix to a tensor
    transform = transforms.ToTensor()

    # Create environment
    env = gym.make("bomberman_rl:bomberman-default-v0")

    # Create agent
    dqn_agent = DQNAgent(11, 13, 5, lr=args.lr)

    # Creating the memory
    memory = ReplayMemory(args.memory_size)

    # Create episodes and call training
    print("Beginning training . . . ")

    for i_episode in range(args.nb_episodes):
        state = env.reset()
        done = False
        loss = -1
        repetitions = 0

        while not done and repetitions < 50:
            # Select and perform an action
            action = dqn_agent.select_action(transform(state).unsqueeze(0).type(torch.FloatTensor).to(dqn_agent.device),
                torch.tensor([dqn_agent.time], device=dqn_agent.device), i_episode, eps_decay=args.nb_episodes)

            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=dqn_agent.device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, dqn_agent.time)

            # Next state
            state = next_state

            if len(memory) >= args.batch_size:
                sample = memory.sample(args.batch_size)
                batch = Simulation(*zip(*sample))
                loss = dqn_agent.train(batch, args.batch_size)

            repetitions += 1

            if args.verbose == True:
                print('Episode[{}/{}], Loss: {:.4f}, Buffer state[{}/{}], Reward: {}'.format(i_episode+1, 
                    args.nb_episodes, loss, len(memory), args.memory_size, reward.item()))

        # Change time to 0 again
        dqn_agent.time = 0

        # Update or not the targetNet
        if (i_episode + 1) % args.target_update == 0: 
            dqn_agent.targetNet.load_state_dict(dqn_agent.qNet.state_dict())


    print("Ended training . . . ")

    if args.save_model != None:
        print("Saving q model . . .")
        net_path = args.save_model + "qNet_at_epoch_" + str(args.nb_episodes) + ".pt"
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        torch.save({'nb_episodes': args.nb_episodes,
                    'qNet': dqn_agent.qNet.state_dict(),
                    'opt_qNet': dqn_agent.optimizer,
                    'memory': memory,
        }, net_path)

        print("Successfully saved")

    # Evaluate agent
    print("Evaluating agent . . . ")

    env_eval = gym.make("bomberman_rl:bomberman-default-v0")
    state = env_eval.reset()
    env_eval.render()
    dqn_agent.qNet.eval()
    done = False

    # Select and perform an action
    while not done:
        action = dqn_agent.select_action(transform(state).unsqueeze(0).type(torch.FloatTensor).to(dqn_agent.device),
            torch.tensor([dqn_agent.time], device=dqn_agent.device), args.nb_episodes, eps_decay=args.nb_episodes)

        next_state, reward, done, _ = env_eval.step(action.item())
        reward = torch.tensor([reward], device=dqn_agent.device)

        # Next state
        state = next_state

        env_eval.render()

    print("Finished!")

if __name__ == '__main__':
    main()