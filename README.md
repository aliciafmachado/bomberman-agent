# Agent and environment to apply RL in the bomberman game

### Install procedure

```
pip install -e .
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
