package gym

import (
	"context"

	"github.com/qntx/gym/rand"
	"github.com/qntx/gym/space"
)

// Info represents auxiliary diagnostic information that is JSON-serializable.
//
// This contains helpful information for debugging, learning, and logging that might include:
// metrics that describe the agent's performance state, variables that are hidden from observations,
// or individual reward terms that are combined to produce the total reward.
type Info map[string]any

// RenderFrame represents a render output which can be various types depending on the render mode.
//
// Examples include:
//   - []byte for RGB array data
//   - string for ANSI text representation
//   - any other format specific to the environment's rendering needs
type RenderFrame any

// Metadata represents environment metadata containing configuration and capability information.
//
// Common metadata includes:
//   - "render_modes": slice of supported render modes (e.g., []string{"human", "rgb_array"})
//   - "render_fps": target frames per second for rendering
//   - "jax": boolean indicating Jax compatibility
//   - "torch": boolean indicating PyTorch compatibility
type Metadata map[string]any

// Env is the main interface for implementing Reinforcement Learning environments.
//
// The interface encapsulates an environment with arbitrary behind-the-scenes dynamics through the
// Step and Reset methods. An environment can be partially or fully observed by single agents.
// For multi-agent environments, consider using specialized multi-agent frameworks.
//
// The main API methods that users of this interface need to know are:
//
//   - Step - Updates an environment with actions returning the next agent observation, the reward for taking that action,
//     if the environment has terminated or truncated due to the latest action and information from the environment about the step.
//   - Reset - Resets the environment to an initial state, required before calling Step.
//     Returns the first agent observation for an episode and information.
//   - Render - Renders the environment to help visualize what the agent sees, examples modes are "human", "rgb_array", "ansi" for text.
//   - Close - Closes the environment, important when external software is used, i.e. graphics libraries for rendering, databases.
//
// Environments have additional methods for users to understand the implementation:
//
//   - ActionSpace - The Space object corresponding to valid actions, all valid actions should be contained within the space.
//   - ObservationSpace - The Space object corresponding to valid observations, all valid observations should be contained within the space.
//   - Spec - An environment spec that contains the information used to initialize the environment.
//   - Metadata - The metadata of the environment, e.g. supported render modes and fps.
//   - GetRNG - The random number generator for the environment for reproducible sampling.
//
// Note: For strict type checking, Env is a generic interface with two parameterized types: Obs and Act.
// The Obs and Act are the expected types of the observations and actions used in Reset and Step.
// The environment's ObservationSpace and ActionSpace should have type Space[Obs] and Space[Act] respectively.
//
// Note: To get reproducible sampling of actions, a seed can be set through the environment's ActionSpace().Seed() method.
type Env[Obs any, Act any] interface {
	// Step runs one timestep of the environment's dynamics using the agent action.
	//
	// When the end of an episode is reached (terminated or truncated), it is necessary to call Reset
	// to reset this environment's state for the next episode.
	//
	// The Step API uses terminated and truncated flags to make it clearer to users when the environment
	// had terminated or truncated, which is critical for reinforcement learning bootstrapping algorithms.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeouts
	//   - action: An action provided by the agent to update the environment state
	//
	// Returns:
	//   - observation: An element of the environment's ObservationSpace as the next observation due to the agent action
	//   - reward: The reward as a result of taking the action
	//   - terminated: Whether the agent reaches the terminal state (as defined under the MDP of the task)
	//     which can be positive or negative. If true, the user needs to call Reset.
	//   - truncated: Whether the truncation condition outside the scope of the MDP is satisfied.
	//     Typically a timelimit, but could also indicate an agent physically going out of bounds.
	//     If true, the user needs to call Reset.
	//   - info: Contains auxiliary diagnostic information (helpful for debugging, learning, and logging)
	//   - error: Any error that occurred during the step
	Step(ctx context.Context, action Act) (Obs, float64, bool, bool, Info, error)

	// Reset resets the environment to an initial internal state, returning an initial observation and info.
	//
	// This method generates a new starting state often with some randomness to ensure that the agent explores the
	// state space and learns a generalized policy about the environment. This randomness can be controlled
	// with the seed parameter; if seed is nil and the environment already has a random number generator,
	// the RNG is not reset.
	//
	// Therefore, Reset should (in the typical use case) be called with a seed right after initialization and then never again.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeouts
	//   - seed: The seed that is used to initialize the environment's RNG. If nil, existing RNG state is preserved.
	//     If provided, the RNG will be reset even if it already exists.
	//   - options: Additional information to specify how the environment is reset (optional, depending on the specific environment)
	//
	// Returns:
	//   - observation: Observation of the initial state. This will be an element of ObservationSpace
	//   - info: Dictionary containing auxiliary information complementing the observation
	//   - error: Any error that occurred during reset
	Reset(ctx context.Context, seed int64, options Info) (Obs, Info, error)

	// Render computes the render frames as specified by the environment's render mode.
	//
	// The environment's Metadata should contain the possible ways to implement the render modes
	// in the "render_modes" key. Common render modes include:
	//
	//   - "human": The environment is continuously rendered for human consumption
	//   - "rgb_array": Return a frame representing the current state as RGB array data
	//   - "ansi": Return a string containing a terminal-style text representation
	//
	// Returns:
	//   - A render frame in the format specified by the environment's render mode
	//   - An error if rendering fails
	Render() (RenderFrame, error)

	// Close performs cleanup when the user has finished using the environment.
	//
	// This is critical for closing rendering windows, database or HTTP connections.
	// Calling Close on an already closed environment should have no effect and not raise an error.
	//
	// Returns:
	//   - An error if cleanup fails
	Close() error

	// ActionSpace returns the Space object corresponding to valid actions.
	//
	// All valid actions should be contained within this space.
	//
	// Returns:
	//   - The action space for this environment
	ActionSpace() space.Space[Act]

	// ObservationSpace returns the Space object corresponding to valid observations.
	//
	// All valid observations should be contained within this space.
	//
	// Returns:
	//   - The observation space for this environment
	ObservationSpace() space.Space[Obs]

	// Metadata returns the metadata of the environment.
	//
	// Common metadata includes render modes, render fps, and framework compatibility flags.
	//
	// Returns:
	//   - A map containing environment metadata
	Metadata() Metadata

	// Spec returns the environment specification used for registration and identification.
	//
	// Returns:
	//   - The environment spec, or nil if not registered
	Spec() *EnvSpec[Obs, Act]

	// Unwrapped returns the base non-wrapped environment.
	//
	// This is useful for accessing the underlying environment when wrappers are applied.
	//
	// Returns:
	//   - The base non-wrapped environment instance
	Unwrapped() Env[Obs, Act]

	// GetRNG returns the environment's random number generator.
	//
	// This RNG is used for reproducible sampling and should be seeded appropriately
	// for deterministic behavior across episodes.
	//
	// Returns:
	//   - The environment's random number generator
	GetRNG() *rand.RNG
}
