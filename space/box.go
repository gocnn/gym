package space

import (
	"fmt"
	"math"

	"github.com/gocnn/gym/rand"
)

// Box represents a (possibly unbounded) box in R^n.
//
// Specifically, a Box represents the Cartesian product of n closed intervals.
// Each interval has the form of one of [a, b], (-∞, b], [a, ∞), or (-∞, ∞).
//
// There are two common use cases:
//
//   - Identical bound for each dimension:
//     Box(low=-1.0, high=2.0, shape=[3, 4])
//
//   - Independent bound for each dimension:
//     Box(low=[-1.0, -2.0], high=[2.0, 4.0])
type Box struct {
	low          []float64 // Lower bounds of the intervals
	high         []float64 // Upper bounds of the intervals
	shape        []int     // Shape of the space
	boundedBelow []bool    // Whether each dimension is bounded below
	boundedAbove []bool    // Whether each dimension is bounded above
	rng          *rand.RNG
}

// NewBox creates a new Box space.
//
// The argument low specifies the lower bound of each dimension and high specifies the upper bounds.
// The space that is constructed will be the product of the intervals [low[i], high[i]].
//
// If low (or high) is a scalar, the lower bound (or upper bound, respectively) will be assumed to be
// this value across all dimensions.
//
// Parameters:
//   - low: Lower bounds of the intervals. Can be a single value or slice
//   - high: Upper bounds of the intervals. Can be a single value or slice
//   - shape: Optional shape specification. If not provided, inferred from low/high
//
// Returns:
//   - A new Box space
//   - An error if the parameters are invalid
func NewBox(low, high interface{}, shape ...[]int) (*Box, error) {
	var lowVec, highVec []float64
	var boxShape []int

	// Parse low parameter
	switch l := low.(type) {
	case float64:
		if len(shape) == 0 {
			return nil, fmt.Errorf("shape must be provided when low is scalar")
		}
		boxShape = shape[0]
		size := 1
		for _, dim := range boxShape {
			size *= dim
		}
		lowVec = make([]float64, size)
		for i := range lowVec {
			lowVec[i] = l
		}
	case []float64:
		lowVec = make([]float64, len(l))
		copy(lowVec, l)
		if len(shape) == 0 {
			boxShape = []int{len(l)}
		} else {
			boxShape = shape[0]
		}
	default:
		return nil, fmt.Errorf("low must be float64 or []float64, got %T", low)
	}

	// Parse high parameter
	switch h := high.(type) {
	case float64:
		if len(lowVec) == 0 {
			return nil, fmt.Errorf("cannot determine size from high when low is not provided")
		}
		highVec = make([]float64, len(lowVec))
		for i := range highVec {
			highVec[i] = h
		}
	case []float64:
		if len(lowVec) != len(h) {
			return nil, fmt.Errorf("low and high must have same length, got %d and %d", len(lowVec), len(h))
		}
		highVec = make([]float64, len(h))
		copy(highVec, h)
	default:
		return nil, fmt.Errorf("high must be float64 or []float64, got %T", high)
	}

	// Validate that low <= high
	for i := range lowVec {
		if lowVec[i] > highVec[i] {
			return nil, fmt.Errorf("low[%d] (%f) must be <= high[%d] (%f)", i, lowVec[i], i, highVec[i])
		}
	}

	// Calculate boundedness
	boundedBelow := make([]bool, len(lowVec))
	boundedAbove := make([]bool, len(lowVec))
	for i := range lowVec {
		boundedBelow[i] = !math.IsInf(lowVec[i], -1)
		boundedAbove[i] = !math.IsInf(highVec[i], 1)
	}

	rng := rand.GetDefaultRNG()

	return &Box{
		low:          lowVec,
		high:         highVec,
		shape:        boxShape,
		boundedBelow: boundedBelow,
		boundedAbove: boundedAbove,
		rng:          rng,
	}, nil
}

// Sample generates a single random sample inside the Box.
//
// In creating a sample of the box, each coordinate is sampled (independently) from a distribution
// that is chosen according to the form of the interval:
//
// * [a, b] : uniform distribution
// * [a, ∞) : shifted exponential distribution
// * (-∞, b] : shifted negative exponential distribution
// * (-∞, ∞) : normal distribution
//
// Parameters:
//   - mask: A mask for sampling values (currently not implemented)
//   - probability: A probability mask for sampling values (currently not implemented)
//
// Returns:
//   - A sampled value from the Box
//   - An error if sampling fails
func (b *Box) Sample(mask any, probability any) ([]float64, error) {
	if mask != nil || probability != nil {
		return nil, fmt.Errorf("mask and probability sampling not yet implemented")
	}

	sample := make([]float64, len(b.low))

	for i := range sample {
		unbounded := !b.boundedBelow[i] && !b.boundedAbove[i]
		uppBounded := !b.boundedBelow[i] && b.boundedAbove[i]
		lowBounded := b.boundedBelow[i] && !b.boundedAbove[i]
		bounded := b.boundedBelow[i] && b.boundedAbove[i]

		switch {
		case unbounded:
			// Normal distribution for unbounded intervals
			sample[i] = b.rng.NormFloat64()
		case lowBounded:
			// Exponential distribution shifted by low bound
			sample[i] = b.rng.ExpFloat64() + b.low[i]
		case uppBounded:
			// Negative exponential distribution shifted by high bound
			sample[i] = b.high[i] - b.rng.ExpFloat64()
		case bounded:
			// Uniform distribution for bounded intervals
			sample[i] = b.low[i] + b.rng.Float64()*(b.high[i]-b.low[i])
		}
	}

	return sample, nil
}

// Seed sets the pseudorandom number generator seed of this space.
//
// Parameters:
//   - seed: The seed value for the space
//
// Returns:
//   - The effective seed value used
//   - An error if seeding fails
func (b *Box) Seed(seed int64) (int64, error) {
	return b.rng.Seed(seed)
}

// Contains returns true if x is a valid member of this space.
//
// Parameters:
//   - x: The element to check for membership
//
// Returns:
//   - true if x is within the bounds of this space, false otherwise
func (b *Box) Contains(x []float64) bool {
	if len(x) != len(b.low) {
		return false
	}

	for i, val := range x {
		if val < b.low[i] || val > b.high[i] {
			return false
		}
	}
	return true
}

// Shape returns the shape of the space elements.
//
// Returns:
//   - A slice of integers representing the shape
func (b *Box) Shape() []int {
	result := make([]int, len(b.shape))
	copy(result, b.shape)
	return result
}

// DType returns the data type of the space elements.
//
// Returns:
//   - "float64" as the data type string
func (b *Box) DType() string {
	return "float64"
}

// IsFlattenable returns true if this space can be flattened to a Box space.
//
// Returns:
//   - true (Box spaces are already flat)
func (b *Box) IsFlattenable() bool {
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
func (b *Box) ToJSONable(samples [][]float64) ([]any, error) {
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
//   - A slice of samples of type []float64
//   - An error if conversion fails or the data is invalid for this space
func (b *Box) FromJSONable(json []any) ([][]float64, error) {
	result := make([][]float64, len(json))
	for i, val := range json {
		switch v := val.(type) {
		case []float64:
			result[i] = v
		case []interface{}:
			floatSlice := make([]float64, len(v))
			for j, elem := range v {
				switch e := elem.(type) {
				case float64:
					floatSlice[j] = e
				case int:
					floatSlice[j] = float64(e)
				default:
					return nil, fmt.Errorf("expected float64 or int, got %T", elem)
				}
			}
			result[i] = floatSlice
		default:
			return nil, fmt.Errorf("expected []float64 or []interface{}, got %T", val)
		}
	}
	return result, nil
}

// String returns a string representation of this space.
//
// Returns:
//   - A string representation showing bounds, shape and dtype
func (b *Box) String() string {
	return fmt.Sprintf("Box(low=%v, high=%v, shape=%v, dtype=float64)", b.low, b.high, b.shape)
}

// IsBounded checks whether the box is bounded in some sense.
//
// Parameters:
//   - manner: One of "both", "below", "above"
//
// Returns:
//   - true if the space is bounded in the specified manner
//   - An error if manner is invalid
func (b *Box) IsBounded(manner string) (bool, error) {
	switch manner {
	case "both":
		for i := range b.low {
			if !b.boundedBelow[i] || !b.boundedAbove[i] {
				return false, nil
			}
		}
		return true, nil
	case "below":
		for i := range b.low {
			if !b.boundedBelow[i] {
				return false, nil
			}
		}
		return true, nil
	case "above":
		for i := range b.low {
			if !b.boundedAbove[i] {
				return false, nil
			}
		}
		return true, nil
	default:
		return false, fmt.Errorf("manner must be one of 'both', 'below', 'above', got '%s'", manner)
	}
}

// Low returns a copy of the lower bounds.
//
// Returns:
//   - A copy of the lower bounds slice
func (b *Box) Low() []float64 {
	result := make([]float64, len(b.low))
	copy(result, b.low)
	return result
}

// High returns a copy of the upper bounds.
//
// Returns:
//   - A copy of the upper bounds slice
func (b *Box) High() []float64 {
	result := make([]float64, len(b.high))
	copy(result, b.high)
	return result
}
