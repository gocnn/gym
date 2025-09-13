# Gym for Go

A lightweight, pure Go implementation of `OpenAI's` Gym for reinforcement learning environments.

## Install

```sh
go get github.com/qntx/gym
```

## Usage

Run a `CartPole-v1` environment with minimal code.

```go
env, err := gym.Make[[]float64, int]("CartPole-v1", map[string]any{"render_mode": "human"})
env.Close()

obs, info, err := env.Reset(context.Background(), 0, nil)

for step := range 500 {
 action, err := env.ActionSpace().Sample(nil, nil)
 obs, reward, done, truncated, info, err := env.Step(context.Background(), action)

 env.Render()

 if done || truncated {
  break
 }
}
```

## Acknowledgments

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/)
