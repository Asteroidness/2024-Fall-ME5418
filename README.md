# RWARE Project

## Overview
This project is an implementation of a reinforcement learning agent using the Multi-Robot Warehouse (RWARE) environment. The agent is designed to solve tasks involving the coordination of multiple robots in a simulated warehouse environment. The implemented method uses Proximal Policy Optimization (PPO) with custom feature extraction, joint action space wrapper, and learning rate scheduling to handle the complex interactions within this multi-agent setting.

## Files
- `RWARE_Final.ipynb`: Jupyter Notebook containing the final implementation, including the setup, training, and evaluation of the reinforcement learning agents.
- `environment.yml`: The Conda environment file listing all the necessary dependencies and versions for this project.

## Requirements
To run the project, you need to install all the required packages listed in the `environment.yml` file. These include packages for machine learning (e.g., `torch`), environment simulation (e.g., `gymnasium`), and visualization (e.g., `matplotlib`).

## Setting Up the Environment
1. Clone the repository to your local machine.
2. Use Conda to create the environment with the following command:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate rware_env
   ```

## Running the Project
Open the Jupyter Notebook (`RWARE_Final.ipynb`) in your preferred editor (e.g., Jupyter Lab or Jupyter Notebook) and run the cells sequentially. The notebook will guide you through:
- Environment setup
- Training the reinforcement learning agent
- Evaluation and visualization of the results

## Key Features
- **Multi-Agent PPO:** Implementation of a decentralized Proximal Policy Optimization algorithm for controlling multiple robots.
- **Custom Feature Extractor:** A tailored feature extractor designed for the RWARE environment, allowing the agents to effectively process the state observations.
- **Joint Action Space Wrapper:** A wrapper to combine multiple agents' observations and actions into joint spaces so that the PPO model can treat the multi-agent system as a single-agent problem.
- **Learning Rate Scheduler:** Integrated ReduceLROnPlateau scheduler to improve training stability by adjusting the learning rate dynamically.

## Custom Components
### MultiAgentFeatureExtractor
The `MultiAgentFeatureExtractor` processes each agent's observations and feeds a unified representation into the PPO model. It consists of two fully connected layers:
```python
class MultiAgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_agents, features_dim):
        self.fc1 = nn.Linear(input_dim_per_agent, 256)
        self.fc2 = nn.Linear(256, features_dim)
    def forward(self, observations):
        x = F.relu(self.fc1(observations))
        x = self.fc2(x)
        return x
```

### CustomPPO
The `CustomPPO` class extends the default PPO from stable-baselines3. It introduces a custom loss function that balances policy loss, value loss, and entropy loss to encourage both exploitation and exploration:
```python
class CustomPPO(PPO):
    def train(self):
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            observations = rollout_data.observations
            actions = rollout_data.actions
            old_log_prob = rollout_data.old_log_prob 
            advantages = rollout_data.advantages
            returns = rollout_data.returns

            values, log_probs, entropy = self.policy.forward(observations)

            ratio = torch.exp(log_probs - old_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range(1.0), 1 + self.clip_range(1.0)) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns.unsqueeze(-1).expand_as(values), values)
            entropy_loss = -entropy.mean()
            custom_loss = self.custom_loss(policy_loss, value_loss, entropy_loss)

            self.optimizer.zero_grad()
            custom_loss.backward()
            self.optimizer.step()

    def custom_loss(self, policy_loss, value_loss, entropy_loss):
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
```

### JointActionSpaceWrapper
The `JointActionSpaceWrapper` combines the actions and observations of all agents into joint spaces so that the PPO model can treat the multi-agent system as a single-agent problem:
```python
class JointActionSpaceWrapper(gym.Env):
    def reset(self):
        obss = self.env.reset()
        return np.concatenate(obss)
    
    def step(self, actions):
        split_actions = np.split(actions, self.n_agents)
        obss, rewards, done, truncated, info = self.env.step(split_actions)
        joint_obs = np.concatenate(obss)
        joint_reward = sum(rewards) / self.n_agents
        joint_done = all(done)
        return joint_obs, joint_reward, joint_done, info
```

## Results
The trained agents showed some capacity for task execution, but overall performance was inconsistent, with fluctuating rewards and instability in training metrics such as value and policy loss.

### Visualizations
- **Learning Rate Schedule**: The learning rate was dynamically adjusted during training using the ReduceLROnPlateau scheduler. The figure below shows the decay in the learning rate over training steps.
- **Training Losses**: The policy loss, value loss, and entropy loss were tracked over training. The figures below illustrate the instability observed during training.
- **Cumulative Rewards**: The cumulative rewards collected by agents were monitored across episodes, as shown in the figure below. Despite some positive trends, the rewards showed high variance and frequent dips, indicating unstable learning.

## Future Work
- **Reward Shaping**: Improve reward shaping to encourage better coordination between robots, such as intermediate rewards for partial task completions or reducing collisions.
- **Network Redesign**: Introduce more advanced neural architectures like Graph Neural Networks (GNNs) or transformer-based models to handle inter-agent relationships effectively.
- **Debugging Tools**: Fix environment rendering issues and add tools for real-time debugging to better understand agent behavior during training.
- **Communication Mechanisms**: Allow robots to exchange limited information to improve coordination and task efficiency.
- **Scalability Testing**: Begin with fewer robots and simpler tasks to ensure a stable foundation before scaling up the complexity of the environment.

## References
- **RWARE Environment**: A multi-robot warehouse environment designed for reinforcement learning research.
- **Stable Baselines3**: A library providing implementations of popular RL algorithms, used here for the PPO implementation.

## License
This project is licensed under the MIT License.

