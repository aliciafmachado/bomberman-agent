# Agent and environment to apply RL in the bomberman game

![](src/asserts/video.gif)

### Install procedure

Make sure that you are in the same folder as the `setup.py`

```
pip install -e .
```

### Demos

##### Run pre-trained agents

In this demo you'll run our pre-trained agents.

First, go to the scripts folder:

```
cd src/scripts
```

Then, run one of the agents:

```
python simulate_single_agent.py --agent-name qagent
python simulate_single_agent.py --agent-name dqn_agent
```

##### Train agents from scratch

In this demo you'll train a Q-learning agent to destroy blocks.

First, go to the scripts folder:

```
cd src/scripts
```

Then, train one of the agents:

```
python train_q_agent.py --agent-name qagent_new
python train_dqn_agent.py --agent-name dqn_agent_new
```

It might be needed to train a bit more the agent, you can
continue training with one of these commands:

```
python train_q_agent.py --agent-name qagent_new --agent-pretrained-name qagent_new
python train_dqn_agent.py --agent-name dqn_agent_new --agent-pretrained-name dqn_agent_new
```

### Run tests

```
python setup.py pytest
```
