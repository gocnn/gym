package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/gocnn/gym/envs/classic"
)

func main() {
	// Direct construction - type-safe and idiomatic Go
	env, err := classic.NewCartPoleEnv(&classic.CartPoleConfig{
		RenderMode: "human",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer env.Close()

	obs, _, err := env.Reset(context.Background(), 0, nil)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Initial: [%.2f, %.2f, %.2f, %.2f]\n", obs[0], obs[1], obs[2], obs[3])

	for step := range 500 {
		action, _ := env.ActionSpace().Sample(nil, nil)
		obs, reward, done, truncated, _, err := env.Step(context.Background(), action)
		if err != nil {
			log.Fatal(err)
		}

		_, err = env.Render()
		if err != nil {
			log.Printf("Render error: %v", err)
		}
		time.Sleep(50 * time.Millisecond)

		if step%50 == 0 {
			fmt.Printf("Step %d: action=%d, reward=%.0f, obs=[%.2f, %.2f, %.2f, %.2f]\n",
				step, action, reward, obs[0], obs[1], obs[2], obs[3])
		}

		if done || truncated {
			fmt.Printf("Episode ended at step %d\n", step)
			break
		}
	}
}
