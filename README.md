# Multi-Agent Reinforcement Learning with PPO

This repository contains a custom implementation of a Proximal Policy Optimization (PPO) model for multi-agent reinforcement learning using the `rware` warehouse environment. It includes a custom feature extractor, joint action space wrapper, and a modified PPO training algorithm to handle multi-agent cooperation.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Custom Components](#custom-components)
  - [MultiAgentFeatureExtractor](#multiagentfeatureextractor)
  - [CustomPPO](#customppo)
  - [JointActionSpaceWrapper](#jointactionspacewrapper)
- [Results](#results)
- [License](#license)

## Overview

In this project, we train multiple agents in a simulated warehouse (`rware-tiny-2ag-v2` environment) to coordinate their actions using the PPO algorithm. The main goal is to optimize the agents' cooperation for completing tasks such as moving and delivering goods.

### Features:
- **Custom Feature Extraction**: The `MultiAgentFeatureExtractor` processes each agent's observations and feeds a unified representation into the PPO model.
- **Custom PPO Model**: The `CustomPPO` class introduces a custom loss function that balances policy and value learning while encouraging exploration.
- **Joint Action Space Wrapper**: The `JointActionSpaceWrapper` combines multiple agents' observations and actions into a joint space, allowing PPO to treat the system as a single agent with a composite observation and action space.

## Requirements

- Python 3.8+
- `stable-baselines3`
- `gymnasium`
- `torch`
- `numpy`


## Usage

### Training

1. **Run the training script**:
    The script initializes the custom PPO model and trains it using joint observations and actions from the multi-agent warehouse environment.

    ```bash
    python train.py
    ```

    The model will be saved after training completes.

2. **Model Configuration**:
    In the script, the PPO model is initialized with custom features and architecture. You can adjust the following parameters in `train.py` to experiment with different architectures:
    - `features_extractor_class`
    - `policy_kwargs`
    - `total_timesteps`

### Evaluation

1. **Load the trained model**:
    To evaluate the trained model, you can load it from the saved file:

    ```bash
    model = PPO.load("ppo_multi_agent_coordinated")
    ```

2. **Run the evaluation loop**:
    The script includes a loop to step through the environment and render the agents' behavior using the trained policy. You can visualize the agents in the warehouse environment as they interact and cooperate.

    ```bash
    python evaluate.py
    ```

### Results
During training, the model's performance can be monitored through logs showing metrics such as:

- `policy_gradient_loss`
- `value_loss`
- `entropy_loss`
- `explained_variance`

The model's behavior improves over time, leading to more coordinated actions between agents in the warehouse environment.

### License
This project is licensed under the MIT License.

