// Package space provides implementations of various space types for reinforcement learning environments.
package space

import (
	"fmt"

	"github.com/gocnn/gym/rand"
)

// Discrete represents a space consisting of finitely many elements.
//
// This class represents a finite subset of integers, more specifically a set of the form {a, a+1, ..., a+n-1}.
//
// Example:
//   - Discrete(2) represents {0, 1}
//   - Discrete(3, start=-1) represents {-1, 0, 1}
type Discrete struct {
	n     int64 // The number of elements of this space
	start int64 // The smallest element of this space
	rng   *rand.RNG
}

// NewDiscrete creates a new Discrete space.
//
// This will construct the space {start, ..., start + n - 1}.
//
// Parameters:
//   - n: The number of elements of this space (must be positive)
//   - start: The smallest element of this space (optional, defaults to 0)
//
// Returns:
//   - A new Discrete space
//   - An error if n is not positive
func NewDiscrete(n int, start ...int) (*Discrete, error) {
	if n <= 0 {
		return nil, fmt.Errorf("n (counts) have to be positive, got %d", n)
	}

	startVal := 0
	if len(start) > 0 {
		startVal = start[0]
	}

	rng := rand.GetDefaultRNG()

	return &Discrete{
		n:     int64(n),
		start: int64(startVal),
		rng:   rng,
	}, nil
}

// Sample generates a single random sample from this space.
//
// A sample will be chosen uniformly at random with the mask if provided,
// or it will be chosen according to a specified probability distribution if the probability mask is provided.
//
// Parameters:
//   - mask: An optional mask for if an action can be selected (currently not implemented)
//   - probability: An optional probability mask (currently not implemented)
//
// Returns:
//   - A sampled integer from the space
//   - An error if sampling fails
func (d *Discrete) Sample(mask any, probability any) (int, error) {
	// TODO: Implement mask and probability sampling
	if mask != nil || probability != nil {
		return 0, fmt.Errorf("mask and probability sampling not yet implemented")
	}

	// Uniform sampling
	sample := d.rng.Int64N(d.n)
	return int(d.start + sample), nil
}

// Seed sets the pseudorandom number generator seed of this space.
//
// Parameters:
//   - seed: The seed value for the space
//
// Returns:
//   - The effective seed value used
//   - An error if seeding fails
func (d *Discrete) Seed(seed int64) (int64, error) {
	return d.rng.Seed(seed)
}

// Contains returns true if x is a valid member of this space.
//
// Parameters:
//   - x: The element to check for membership
//
// Returns:
//   - true if x is in the range [start, start + n), false otherwise
func (d *Discrete) Contains(x int) bool {
	return x >= int(d.start) && x < int(d.start+d.n)
}

// Shape returns the shape of the space elements.
//
// Discrete spaces don't have a well-defined shape, so this returns nil.
//
// Returns:
//   - nil (discrete spaces are scalar)
func (d *Discrete) Shape() []int {
	return nil
}

// DType returns the data type of the space elements.
//
// Returns:
//   - "int64" as the data type string
func (d *Discrete) DType() string {
	return "int64"
}

// IsFlattenable returns true if this space can be flattened to a Box space.
//
// Returns:
//   - true (discrete spaces can be flattened)
func (d *Discrete) IsFlattenable() bool {
	return true
}

// ToJSONable converts a batch of samples from this space to a JSONable data type.
//
// Parameters:
//   - samples: A slice of samples from this space
//
// Returns:
//   - A slice of any type that can be marshaled to JSON
//   - An error if conversion fails
func (d *Discrete) ToJSONable(samples []int) ([]any, error) {
	result := make([]any, len(samples))
	for i, sample := range samples {
		result[i] = sample
	}
	return result, nil
}

// FromJSONable converts a JSONable data type to a batch of samples from this space.
//
// Parameters:
//   - json: A slice of any type that was previously created by ToJSONable
//
// Returns:
//   - A slice of samples of type int
//   - An error if conversion fails or the data is invalid for this space
func (d *Discrete) FromJSONable(json []any) ([]int, error) {
	result := make([]int, len(json))
	for i, val := range json {
		switch v := val.(type) {
		case int:
			result[i] = v
		case float64:
			result[i] = int(v)
		case int64:
			result[i] = int(v)
		default:
			return nil, fmt.Errorf("expected int-like value, got %T", val)
		}
	}
	return result, nil
}

// String returns a string representation of this space.
//
// Returns:
//   - A string representation in the format "Discrete(n)" or "Discrete(n, start=s)"
func (d *Discrete) String() string {
	if d.start != 0 {
		return fmt.Sprintf("Discrete(%d, start=%d)", d.n, d.start)
	}
	return fmt.Sprintf("Discrete(%d)", d.n)
}

// N returns the number of elements in this space.
//
// Returns:
//   - The number of elements (n)
func (d *Discrete) N() int {
	return int(d.n)
}

// Start returns the starting value of this space.
//
// Returns:
//   - The starting value
func (d *Discrete) Start() int {
	return int(d.start)
}
