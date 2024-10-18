# Multi-Robot Warehouse Environment with Reinforcement Learning

This project involves using reinforcement learning to optimize robot navigation and task allocation in a multi-robot warehouse environment (RWARE). The goal is to develop algorithms that enable robots to efficiently deliver requested goods while minimizing congestion and collisions in a simulated warehouse setting.

## Project Background

The Multi-Robot Warehouse (RWARE) environment provides a platform for simulating the behaviors of multiple robots in a warehouse. It involves solving challenges related to path planning, task scheduling, and resource management. This project uses reinforcement learning (RL) approaches to tackle these problems by training robots to make decisions that maximize overall efficiency in task completion.

## Prerequisites and Dependencies

To set up and run this project, the following software dependencies must be installed in the `rl_env` Conda environment:

### Python Packages

| Package                | Version   | Description                                       |
|------------------------|-----------|---------------------------------------------------|
| gymnasium              | 0.28.1    | Provides the environment interfaces for RL.       |
| stable-baselines3      | 2.3.2     | RL algorithms package (PPO, DQN, A2C, etc.).      |
| numpy                  | 1.24.4    | Fundamental package for numerical computations.   |
| pandas                 | 2.0.3     | Data manipulation and analysis library.           |
| matplotlib             | 3.7.3     | Plotting library for visualizing results.         |
| scikit-image           | 0.21.0    | Image processing library for analyzing data.      |
| rware                  | 2.0.0     | Multi-Robot Warehouse environment.                |
| torch (PyTorch)        | 2.4.0     | Deep learning library used for training models.   |
| jupyter                | (latest)  | Interactive notebook environment.                 |
| ray                    | 2.10.0    | Framework for scaling and parallelizing RL tasks. |

### Other Libraries

Additional dependencies like `protobuf`, `dm-tree`, and `cloudpickle` support the main packages and provide enhanced functionality.

## Activate the environment
`conda activate rl_env`

## License
This project is licensed under the MIT License.

## References
Gymnasium Documentation
Stable Baselines3 Documentation
Ray Documentation
RWARE GitHub Repository
