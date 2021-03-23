import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Parsing arguments
parser = argparse.ArgumentParser(description='Plots a pickle file with logs')
parser.add_argument('--file', help="name of the file", required=True)
parser.add_argument('--path', help="path to file with logs", default="./")
parser.add_argument('--window', help="window size for moving average", default=1)

args = parser.parse_args()

with open(args.path+args.file+".pickle", 'rb') as handle:
    rewards = pickle.load(handle)

rewards = np.convolve(np.array(rewards), np.ones(int(args.window)), 'valid') / int(args.window)

plt.plot(rewards)
plt.show()
