// Package classic provides classic control environments for reinforcement learning.
//
// This package implements well-known control problems like CartPole, MountainCar, etc.
// that are commonly used for testing and benchmarking reinforcement learning algorithms.
package classic

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"github.com/qntx/gym"
	"github.com/qntx/gym/rand"
	"github.com/qntx/gym/space"
)

// CartPoleEnv implements the classic cart-pole system described by Rich Sutton et al.
//
// This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
// "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem".
// A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
// The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
// in the left and right direction on the cart.
//
// ## Action Space
// The action is an integer which can take values {0, 1} indicating the direction
// of the fixed force the cart is pushed with.
// - 0: Push cart to the left
// - 1: Push cart to the right
//
// ## Observation Space
// The observation is a 4-element array with the values corresponding to the following positions and velocities:
// | Index | Observation           | Min                 | Max               |
// |-------|-----------------------|---------------------|-------------------|
// | 0     | Cart Position         | -4.8                | 4.8               |
// | 1     | Cart Velocity         | -Inf                | Inf               |
// | 2     | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
// | 3     | Pole Angular Velocity | -Inf                | Inf               |
//
// ## Rewards
// A reward of +1 is given for every step taken, including the termination step.
// If SuttonBartoReward is true, then a reward of 0 is awarded for every non-terminating step
// and -1 for the terminating step.
//
// ## Episode End
// The episode ends if any one of the following occurs:
// 1. Termination: Pole Angle is greater than ±12°
// 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
// 3. Truncation: Episode length is greater than 500 (handled by TimeLimit wrapper)
type CartPoleEnv struct {
	// Environment parameters
	gravity              float64
	masscart             float64
	masspole             float64
	totalMass            float64
	length               float64 // actually half the pole's length
	polemasslength       float64
	forceMag             float64
	tau                  float64 // seconds between state updates
	kinematicsIntegrator string

	// Thresholds
	thetaThresholdRadians float64
	xThreshold            float64

	// State
	state []float64 // [x, x_dot, theta, theta_dot]
	rng   *rand.RNG

	// Configuration
	suttonBartoReward bool
	renderMode        string

	// Spaces
	actionSpace      space.Space[int]
	observationSpace space.Space[[]float64]

	// Episode tracking
	stepsBeyondTerminated *int

	// Metadata and spec
	metadata gym.Metadata
	spec     *gym.EnvSpec[[]float64, int]

	// Rendering
	screen *ebiten.Image

	// Auto-rendering support
	autoRenderGame *AutoRenderGame
	renderMutex    sync.Mutex
}

// CartPoleConfig holds configuration options for CartPole environment
type CartPoleConfig struct {
	SuttonBartoReward bool
	RenderMode        string
}

// NewCartPoleEnv creates a new CartPole environment instance.
//
// Parameters:
//   - config: Configuration options for the environment
//
// Returns:
//   - A new CartPole environment
//   - An error if initialization fails
func NewCartPoleEnv(config *CartPoleConfig) (*CartPoleEnv, error) {
	if config == nil {
		config = &CartPoleConfig{}
	}

	env := &CartPoleEnv{
		// Physics parameters matching Python implementation
		gravity:              9.8,
		masscart:             1.0,
		masspole:             0.1,
		length:               0.5, // actually half the pole's length
		forceMag:             10.0,
		tau:                  0.02, // seconds between state updates
		kinematicsIntegrator: "euler",

		// Thresholds
		thetaThresholdRadians: 12 * 2 * math.Pi / 360, // ±12°
		xThreshold:            2.4,

		// Configuration
		suttonBartoReward: config.SuttonBartoReward,
		renderMode:        config.RenderMode,

		// Metadata and spec
		metadata: gym.Metadata{
			"render_modes": []string{"human", "rgb_array"},
			"render_fps":   50,
		},
		spec: &gym.EnvSpec[[]float64, int]{
			ID: "CartPole-v1",
			EntryPoint: func(config interface{}) (gym.Env[[]float64, int], error) {
				var cartPoleConfig *CartPoleConfig
				if config != nil {
					if configMap, ok := config.(map[string]interface{}); ok {
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

				return NewCartPoleEnv(cartPoleConfig)
			},
			RewardThreshold:   &[]float64{475.0}[0],
			MaxEpisodeSteps:   &[]int{500}[0],
			Nondeterministic:  false,
			OrderEnforce:      true,
			DisableEnvChecker: false,
			Kwargs:            make(map[string]interface{}),
		},
	}

	// Calculate derived parameters
	env.totalMass = env.masspole + env.masscart
	env.polemasslength = env.masspole * env.length

	// Initialize RNG
	rng, _, err := rand.NewRNG(0)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNG: %w", err)
	}
	env.rng = rng

	// Create action space: Discrete(2) for left/right actions
	actionSpace, err := space.NewDiscrete(2)
	if err != nil {
		return nil, fmt.Errorf("failed to create action space: %w", err)
	}
	env.actionSpace = actionSpace

	// Create observation space: Box(4) with bounds
	high := []float64{
		env.xThreshold * 2,
		math.Inf(1),
		env.thetaThresholdRadians * 2,
		math.Inf(1),
	}
	low := []float64{
		-env.xThreshold * 2,
		math.Inf(-1),
		-env.thetaThresholdRadians * 2,
		math.Inf(-1),
	}
	observationSpace, err := space.NewBox(low, high)
	if err != nil {
		return nil, fmt.Errorf("failed to create observation space: %w", err)
	}
	env.observationSpace = observationSpace

	return env, nil
}

// Close performs cleanup when the user has finished using the environment.
func (env *CartPoleEnv) Close() error {
	if env.screen != nil {
		env.screen.Dispose()
		env.screen = nil
	}
	return nil
}

// Step runs one timestep of the environment's dynamics using the agent action.
func (env *CartPoleEnv) Step(ctx context.Context, action int) ([]float64, float64, bool, bool, gym.Info, error) {
	if !env.actionSpace.Contains(action) {
		return nil, 0, false, false, nil, fmt.Errorf("invalid action %d", action)
	}

	if env.state == nil {
		return nil, 0, false, false, nil, fmt.Errorf("call Reset before using Step method")
	}

	x, xDot, theta, thetaDot := env.state[0], env.state[1], env.state[2], env.state[3]

	// Convert action to force
	force := env.forceMag
	if action == 0 {
		force = -env.forceMag
	}

	costheta := math.Cos(theta)
	sintheta := math.Sin(theta)

	// Physics simulation (from the referenced paper)
	temp := (force + env.polemasslength*thetaDot*thetaDot*sintheta) / env.totalMass
	thetaacc := (env.gravity*sintheta - costheta*temp) / (env.length * (4.0/3.0 - env.masspole*costheta*costheta/env.totalMass))
	xacc := temp - env.polemasslength*thetaacc*costheta/env.totalMass

	// Update state using Euler integration
	if env.kinematicsIntegrator == "euler" {
		x = x + env.tau*xDot
		xDot = xDot + env.tau*xacc
		theta = theta + env.tau*thetaDot
		thetaDot = thetaDot + env.tau*thetaacc
	} else { // semi-implicit euler
		xDot = xDot + env.tau*xacc
		x = x + env.tau*xDot
		thetaDot = thetaDot + env.tau*thetaacc
		theta = theta + env.tau*thetaDot
	}

	env.state = []float64{x, xDot, theta, thetaDot}

	// Check termination conditions
	terminated := x < -env.xThreshold ||
		x > env.xThreshold ||
		theta < -env.thetaThresholdRadians ||
		theta > env.thetaThresholdRadians

	var reward float64
	if !terminated {
		if env.suttonBartoReward {
			reward = 0.0
		} else {
			reward = 1.0
		}
	} else if env.stepsBeyondTerminated == nil {
		// Pole just fell!
		env.stepsBeyondTerminated = new(int)
		*env.stepsBeyondTerminated = 0
		if env.suttonBartoReward {
			reward = -1.0
		} else {
			reward = 1.0
		}
	} else {
		if *env.stepsBeyondTerminated == 0 {
			// Log warning about calling step after termination
			fmt.Printf("Warning: calling Step() even though environment has already returned terminated = true\n")
		}
		*env.stepsBeyondTerminated++
		if env.suttonBartoReward {
			reward = -1.0
		} else {
			reward = 0.0
		}
	}

	// Create observation (copy of state as float32 in Python version)
	observation := make([]float64, len(env.state))
	copy(observation, env.state)

	// truncation=false as the time limit is handled by the TimeLimit wrapper
	return observation, reward, terminated, false, gym.Info{}, nil
}

// Reset resets the environment to an initial internal state, returning an initial observation and info.
func (env *CartPoleEnv) Reset(ctx context.Context, seed int64, options gym.Info) ([]float64, gym.Info, error) {
	// Seed the RNG if provided
	if seed != 0 {
		_, err := env.rng.Seed(seed)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to seed RNG: %w", err)
		}
	}

	// Parse reset bounds from options
	low, high := -0.05, 0.05 // default bounds
	if options != nil {
		if lowVal, ok := options["low"].(float64); ok {
			low = lowVal
		}
		if highVal, ok := options["high"].(float64); ok {
			high = highVal
		}
	}

	// Initialize state with uniform random values
	env.state = make([]float64, 4)
	for i := range env.state {
		env.state[i] = low + env.rng.Float64()*(high-low)
	}

	env.stepsBeyondTerminated = nil

	// Create observation (copy of state)
	observation := make([]float64, len(env.state))
	copy(observation, env.state)

	return observation, gym.Info{}, nil
}

// Render computes the render frames as specified by the environment's render mode.
func (env *CartPoleEnv) Render() (gym.RenderFrame, error) {
	if env.renderMode == "" {
		return nil, fmt.Errorf("no render mode specified")
	}

	if env.state == nil {
		return nil, fmt.Errorf("environment state is nil, call Reset first")
	}

	env.renderMutex.Lock()
	defer env.renderMutex.Unlock()

	// Initialize screen if not already done
	if env.screen == nil {
		screenWidth, screenHeight := 600, 400
		env.screen = ebiten.NewImage(screenWidth, screenHeight)
	}

	// Clear screen with white background
	env.screen.Fill(color.RGBA{255, 255, 255, 255})

	// Get screen dimensions
	bounds := env.screen.Bounds()
	screenWidth := float64(bounds.Dx())
	screenHeight := float64(bounds.Dy())

	// Calculate scaling and positions
	worldWidth := env.xThreshold * 2 // 4.8
	scale := screenWidth / worldWidth
	cartx := env.state[0]*scale + screenWidth/2
	carty := screenHeight - 100.0 // Position from bottom

	// CartPole parameters
	polelen := scale * (2 * env.length) // use actual length from env
	polewidth := 10.0
	cartwidth, cartheight := 50.0, 30.0
	axleoffset := cartheight / 4.0

	// Draw track (horizontal line)
	vector.StrokeLine(env.screen, 0, float32(carty), float32(screenWidth), float32(carty), 2, color.RGBA{0, 0, 0, 255}, false)

	// Draw cart (rectangle)
	cartLeft := cartx - cartwidth/2
	cartTop := carty - cartheight/2

	// Draw cart as filled rectangle
	vector.DrawFilledRect(env.screen, float32(cartLeft), float32(cartTop), float32(cartwidth), float32(cartheight), color.RGBA{0, 0, 0, 255}, false)

	// Calculate pole position
	theta := env.state[2]
	poleEndX := cartx + math.Sin(theta)*polelen
	poleEndY := carty - math.Cos(theta)*polelen

	// Draw pole (line with thickness)
	vector.StrokeLine(env.screen, float32(cartx), float32(carty-axleoffset), float32(poleEndX), float32(poleEndY), float32(polewidth), color.RGBA{202, 152, 101, 255}, false)

	// Draw axle (circle)
	vector.DrawFilledCircle(env.screen, float32(cartx), float32(carty-axleoffset), float32(polewidth/2), color.RGBA{129, 132, 203, 255}, false)

	// Display debug information
	debugText := "CartPole Environment\n"
	debugText += fmt.Sprintf("Position: %.2f\n", env.state[0])
	debugText += fmt.Sprintf("Velocity: %.2f\n", env.state[1])
	debugText += fmt.Sprintf("Angle: %.2f rad (%.1f°)\n", env.state[2], env.state[2]*180/math.Pi)
	debugText += fmt.Sprintf("Angular Vel: %.2f\n", env.state[3])

	ebitenutil.DebugPrint(env.screen, debugText)

	// Auto-start rendering window for "human" mode
	if env.renderMode == "human" && env.autoRenderGame == nil {
		env.startAutoRender()
	}

	if env.renderMode == "rgb_array" {
		// Convert screen to RGB array
		bounds := env.screen.Bounds()
		rgbaImg := image.NewRGBA(bounds)

		// Read pixels from Ebiten image
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				c := env.screen.At(x, y)
				rgbaImg.Set(x, y, c)
			}
		}

		return rgbaImg, nil
	}

	// For "human" mode, return the Ebiten image directly
	return env.screen, nil
}

// startAutoRender starts the automatic rendering window in a separate goroutine
func (env *CartPoleEnv) startAutoRender() {
	env.autoRenderGame = &AutoRenderGame{
		env:   env,
		mutex: &env.renderMutex,
	}

	go func() {
		ebiten.SetWindowSize(600, 400)
		ebiten.SetWindowTitle("CartPole Environment")
		ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

		// Run the game loop
		if err := ebiten.RunGame(env.autoRenderGame); err != nil {
			// Window was closed, clean up
			env.renderMutex.Lock()
			env.autoRenderGame = nil
			env.renderMutex.Unlock()
		}
	}()

	// Give the window a moment to initialize
	time.Sleep(100 * time.Millisecond)
}

// AutoRenderGame manages automatic rendering window
type AutoRenderGame struct {
	env   *CartPoleEnv
	mutex *sync.Mutex
}

func (g *AutoRenderGame) Update() error {
	return nil // No game logic needed, just display
}

func (g *AutoRenderGame) Draw(screen *ebiten.Image) {
	g.mutex.Lock()
	defer g.mutex.Unlock()

	if g.env.screen != nil {
		screen.DrawImage(g.env.screen, nil)
	}
}

func (g *AutoRenderGame) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 600, 400
}

// ActionSpace returns the Space object corresponding to valid actions.
func (env *CartPoleEnv) ActionSpace() space.Space[int] {
	return env.actionSpace
}

// ObservationSpace returns the Space object corresponding to valid observations.
func (env *CartPoleEnv) ObservationSpace() space.Space[[]float64] {
	return env.observationSpace
}

// Metadata returns the metadata of the environment.
func (env *CartPoleEnv) Metadata() gym.Metadata {
	return env.metadata
}

// Spec returns the environment specification used for registration and identification.
func (env *CartPoleEnv) Spec() *gym.EnvSpec[[]float64, int] {
	return env.spec
}

// Unwrapped returns the base non-wrapped environment.
func (env *CartPoleEnv) Unwrapped() gym.Env[[]float64, int] {
	return env
}

// GetRNG returns the environment's random number generator.
func (env *CartPoleEnv) GetRNG() *rand.RNG {
	return env.rng
}
