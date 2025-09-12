// Package space provides the implementation of the Space interface for defining observation and action spaces.
package space

// Space is a generic interface used to define observation and action spaces in reinforcement learning environments.
//
// Spaces are crucially used in Gym to define the format of valid actions and observations.
// They serve various purposes:
//
//   - They clearly define how to interact with environments, i.e. they specify what actions need to look like
//     and what observations will look like
//   - They allow us to work with highly structured data (e.g. in the form of elements of Dict spaces)
//     and painlessly transform them into flat arrays that can be used in learning code
//   - They provide a method to sample random elements. This is especially useful for exploration and debugging.
//
// Different spaces can be combined hierarchically via container spaces (Tuple and Dict) to build a
// more expressive space.
//
// Warning: Custom observation & action spaces can implement the Space interface. However, most use-cases
// should be covered by the existing space implementations (e.g. Box, Discrete, etc...), and container
// implementations (Tuple & Dict). Note that parametrized probability distributions (through the
// Sample method), and batching functions (in vector environments), are only well-defined for instances
// of spaces provided by default. Moreover, some implementations of Reinforcement Learning algorithms might
// not handle custom spaces properly. Use custom spaces with care.
type Space[T any] interface {
	// Sample randomly samples an element of this space.
	//
	// Can be uniform or non-uniform sampling based on boundedness of space.
	// The binary mask and the probability mask can't be used at the same time.
	//
	// Parameters:
	//   - mask: A mask used for random sampling, expected to be compatible with the space's sample implementation
	//   - probability: A probability mask used for sampling according to the given probability distribution
	//
	// Returns:
	//   - A sampled element from the space
	//   - An error if sampling fails or invalid parameters are provided
	Sample(mask any, probability any) (T, error)

	// Seed sets the pseudorandom number generator (PRNG) seed of this space and, if applicable, the PRNGs of subspaces.
	//
	// Parameters:
	//   - seed: The seed value for the space. This is expanded for composite spaces to accept multiple values.
	//
	// Returns:
	//   - The effective seed value used for the PRNG
	//   - An error if seeding fails
	Seed(seed int64) (int64, error)

	// Contains returns true if x is a valid member of this space, equivalent to checking if sample belongs to space.
	//
	// Parameters:
	//   - x: The element to check for membership in this space
	//
	// Returns:
	//   - true if x is a valid member of this space, false otherwise
	Contains(x T) bool

	// Shape returns the shape of the space elements.
	//
	// If elements of the space are arrays, this should specify their shape.
	// Returns nil if the space doesn't have a well-defined shape.
	//
	// Returns:
	//   - A slice of integers representing the shape, or nil if not applicable
	Shape() []int

	// DType returns the data type of the space elements.
	//
	// If elements of the space are typed arrays, this should specify their data type.
	//
	// Returns:
	//   - A string representation of the data type (e.g., "float64", "int", "bool")
	DType() string

	// IsFlattenable returns true if this space can be flattened to a Box space.
	//
	// This is useful for determining if the space can be converted to a flat array
	// representation for use with learning algorithms that expect vectorized input.
	//
	// Returns:
	//   - true if the space can be flattened, false otherwise
	IsFlattenable() bool

	// ToJSONable converts a batch of samples from this space to a JSONable data type.
	//
	// This method is useful for serialization and communication between different
	// components of a reinforcement learning system.
	//
	// Parameters:
	//   - samples: A slice of samples from this space to convert
	//
	// Returns:
	//   - A slice of any type that can be marshaled to JSON
	//   - An error if conversion fails
	ToJSONable(samples []T) ([]any, error)

	// FromJSONable converts a JSONable data type to a batch of samples from this space.
	//
	// This method is the inverse of ToJSONable and is useful for deserialization.
	//
	// Parameters:
	//   - json: A slice of any type that was previously created by ToJSONable
	//
	// Returns:
	//   - A slice of samples of type T
	//   - An error if conversion fails or the data is invalid for this space
	FromJSONable(json []any) ([]T, error)
}
