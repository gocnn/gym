# Gym for Go

A lightweight, pure Go implementation of `OpenAI's` Gym for reinforcement learning environments.

## Install

```sh
go get github.com/qntx/gym
```

## Usage

Run a `CartPole-v1` environment with minimal code.

```go
env, err := gym.Make[[]float64, int]("CartPole-v1", map[string]any{
    "render_mode": "human",
    "sutton_barto_reward": false,
})
defer env.Close()

obs, info, err := env.Reset(context.Background(), 0, nil)
for step := 0; step < 500; step++ {
    action, err := env.ActionSpace().Sample(nil, nil)
    obs, reward, done, truncated, info, err := env.Step(context.Background(), action)
    env.Render()
    if done || truncated {
        break
    }
}
```

## Environments

This library provides implementations of classic reinforcement learning environments. The table below shows the current implementation status:

| **Category** | **Environment**              | **Rendering** | **Vectorized** | **Configurable** | **Action Space**  | **Observation Space** | **Implemented** |
| ------------ | ---------------------------- | ------------- | -------------- | ---------------- | ----------------- | --------------------- | --------------- |
| Classic      |                              |               |                |                  |                   |                       |                 |
|              | `CartPole-v1`                | Y             | N              | Y                | Discrete(2)       | Box(4,)               | √               |
|              | `CartPole-v0`                | Y             | N              | Y                | Discrete(2)       | Box(4,)               | √               |
|              | `Acrobot-v1`                 | N             | N              | N                | Discrete(3)       | Box(6,)               |                 |
|              | `MountainCar-v0`             | N             | N              | N                | Discrete(3)       | Box(2,)               |                 |
|              | `MountainCarContinuous-v0`   | N             | N              | N                | Box(1,)           | Box(2,)               |                 |
|              | `Pendulum-v1`                | N             | N              | N                | Box(1,)           | Box(3,)               |                 |
| Box2D        |                              |               |                |                  |                   |                       |                 |
|              | `LunarLander-v2`             | N             | N              | N                | Discrete(4)       | Box(8,)               |                 |
|              | `LunarLanderContinuous-v2`   | N             | N              | N                | Box(2,)           | Box(8,)               |                 |
|              | `BipedalWalker-v3`           | N             | N              | N                | Box(4,)           | Box(24,)              |                 |
|              | `BipedalWalkerHardcore-v3`   | N             | N              | N                | Box(4,)           | Box(24,)              |                 |
|              | `CarRacing-v2`               | N             | N              | N                | Box(3,)           | Box(96,96,3)          |                 |
| Toy Text     |                              |               |                |                  |                   |                       |                 |
|              | `FrozenLake-v1`              | N             | N              | N                | Discrete(4)       | Discrete(16)          |                 |
|              | `FrozenLake8x8-v1`           | N             | N              | N                | Discrete(4)       | Discrete(64)          |                 |
|              | `CliffWalking-v0`            | N             | N              | N                | Discrete(4)       | Discrete(48)          |                 |
|              | `Taxi-v3`                    | N             | N              | N                | Discrete(6)       | Discrete(500)         |                 |
|              | `Blackjack-v1`               | N             | N              | N                | Discrete(2)       | Tuple(32,11,2)        |                 |

**Legend:**

- **Rendering**: Supports visual rendering (human mode)
- **Vectorized**: Supports running multiple environments in parallel
- **Configurable**: Supports runtime configuration options
- **Implemented**: √ = Fully implemented, empty = Not implemented

## Acknowledgments

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/)
