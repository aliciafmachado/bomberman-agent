# Agent and environment to apply RL in the bomberman game

### Install procedure

Make sure that you are in the same folder as the `setup.py`

```
pip install -e .
```

### Demos

##### Train and run Q-agent in very small environment

In this demo you'll train a Q-learning agent to destroy all blocks in our smallest environment

First, go to the scripts folder:

```
cd src/scripts
```

Then, train and run the trained agent:

```
python train_q_agent.py --agent-name qagent --environment bomberman_rl:bomberman-minimal-v0
python simulate_single_agent.py --agent-name qagent --environment bomberman_rl:bomberman-minimal-v0
```

### Train a single QAgent

To train the QAgent you should use the script `train_q_agent.py`. First go to the 
script's folder:

```
cd bomberman-agent/src/scripts
```

Now you should run the train:

```
python train_q_agent.py --agent-name qagent
```

It might be needed to train a bit more the agent, you can
continue training with this command:

```
python train_q_agent.py --agent-name qagent --agent-pretrained-name qagent
```

### Run a single Agent

```
python simulate_single_agent.py --agent-name qagent
```

### Run tests

```
python setup.py pytest
```
