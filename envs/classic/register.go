package classic

import (
	"github.com/qntx/gym"
)

func init() {
	// Register CartPole-v1 environment
	gym.Register(
		"CartPole-v1",
		func(config any) (gym.Env[[]float64, int], error) {
			var cartPoleConfig *CartPoleConfig
			if config != nil {
				if configMap, ok := config.(map[string]any); ok {
					cartPoleConfig = &CartPoleConfig{}

					// Parse render_mode
					if renderMode, exists := configMap["render_mode"]; exists {
						if rm, ok := renderMode.(string); ok {
							cartPoleConfig.RenderMode = rm
						}
					}

					// Parse sutton_barto_reward
					if suttonBarto, exists := configMap["sutton_barto_reward"]; exists {
						if sb, ok := suttonBarto.(bool); ok {
							cartPoleConfig.SuttonBartoReward = sb
						}
					}
				}
			}

			env, err := NewCartPoleEnv(cartPoleConfig)
			if err != nil {
				return nil, err
			}

			return env, nil
		},
		gym.WithMaxEpisodeSteps[[]float64, int](500),
		gym.WithRewardThreshold[[]float64, int](475.0),
	)

}
