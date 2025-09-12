// Package rand provides RNG utilities for Gymnasium-Go, based on math/rand/v2.
//
// This package provides seeding, generator, and random number generation functions
// similar to Python Gymnasium's seeding utilities, ensuring reproducible behavior
// across reinforcement learning experiments.
package rand

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"time"
)

// RNG is a seeded random number generator that provides thread-safe random number generation
// with comprehensive utility methods for reinforcement learning environments.
type RNG struct {
	rng  *rand.Rand
	seed int64
	mu   sync.RWMutex
}

// DefaultRNG is a singleton instance for global use
var (
	defaultRNG     *RNG
	defaultRNGOnce sync.Once
)

// NewRNG creates a new RNG instance with the given seed.
//
// If seed is 0, a random seed based on current time will be used.
// This function validates the seed and returns an error for invalid values.
//
// Parameters:
//   - seed: The seed value for the RNG. Must be non-negative.
//
// Returns:
//   - A new RNG instance
//   - The effective seed used (useful when seed=0)
//   - An error if the seed is invalid
func NewRNG(seed int64) (*RNG, int64, error) {
	if seed < 0 {
		return nil, 0, fmt.Errorf("seed must be non-negative, got: %d", seed)
	}

	effectiveSeed := seed
	if seed == 0 {
		effectiveSeed = time.Now().UnixNano()
	}

	source := rand.NewPCG(uint64(effectiveSeed), uint64(effectiveSeed))
	rng := &RNG{
		rng:  rand.New(source),
		seed: effectiveSeed,
	}

	return rng, effectiveSeed, nil
}

// DefaultRNG returns the singleton default RNG instance.
//
// This is thread-safe and will be initialized with a random seed on first access.
//
// Returns:
//   - The default RNG instance
func GetDefaultRNG() *RNG {
	defaultRNGOnce.Do(func() {
		rng, _, _ := NewRNG(0) // Use random seed
		defaultRNG = rng
	})
	return defaultRNG
}

// Seed resets the RNG with a new seed value.
//
// Parameters:
//   - seed: The new seed value. Must be non-negative.
//
// Returns:
//   - The effective seed used
//   - An error if the seed is invalid
func (r *RNG) Seed(seed int64) (int64, error) {
	if seed < 0 {
		return 0, fmt.Errorf("seed must be non-negative, got: %d", seed)
	}

	effectiveSeed := seed
	if seed == 0 {
		effectiveSeed = time.Now().UnixNano()
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	source := rand.NewPCG(uint64(effectiveSeed), uint64(effectiveSeed))
	r.rng = rand.New(source)
	r.seed = effectiveSeed

	return effectiveSeed, nil
}

// GetSeed returns the current seed value.
//
// Returns:
//   - The current seed value
func (r *RNG) GetSeed() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.seed
}

// Int63 returns a non-negative pseudo-random 63-bit integer as an int64.
func (r *RNG) Int63() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Int64() >> 1 // Ensure non-negative by clearing sign bit
}

// Int returns a non-negative pseudo-random int.
func (r *RNG) Int() int {
	return int(r.Int63())
}

// Int64 returns a pseudo-random 64-bit value as an int64.
func (r *RNG) Int64() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Int64()
}

// Uint64 returns a pseudo-random 64-bit value as a uint64.
func (r *RNG) Uint64() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Uint64()
}

// IntN returns, as an int, a non-negative pseudo-random number in the half-open interval [0,n).
// It panics if n <= 0.
func (r *RNG) IntN(n int) int {
	if n <= 0 {
		panic(fmt.Sprintf("invalid argument to IntN: %d", n))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.IntN(n)
}

// Int64N returns, as an int64, a non-negative pseudo-random number in the half-open interval [0,n).
// It panics if n <= 0.
func (r *RNG) Int64N(n int64) int64 {
	if n <= 0 {
		panic(fmt.Sprintf("invalid argument to Int64N: %d", n))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Int64N(n)
}

// Float32 returns, as a float32, a pseudo-random number in the half-open interval [0.0,1.0).
func (r *RNG) Float32() float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Float32()
}

// Float64 returns, as a float64, a pseudo-random number in the half-open interval [0.0,1.0).
func (r *RNG) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Float64()
}

// NormFloat64 returns a normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64]
// with standard normal distribution (mean = 0, stddev = 1).
func (r *RNG) NormFloat64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.NormFloat64()
}

// ExpFloat64 returns an exponentially distributed float64 in the range (0, +math.MaxFloat64]
// with an exponential distribution whose rate parameter (lambda) is 1.
func (r *RNG) ExpFloat64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.ExpFloat64()
}

// Perm returns, as a slice of n ints, a pseudo-random permutation of the integers [0,n).
// It panics if n < 0.
func (r *RNG) Perm(n int) []int {
	if n < 0 {
		panic(fmt.Sprintf("invalid argument to Perm: %d", n))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Perm(n)
}

// Shuffle pseudo-randomizes the order of elements using the Fisher-Yates shuffle algorithm.
// It panics if n < 0.
func (r *RNG) Shuffle(n int, swap func(i, j int)) {
	if n < 0 {
		panic(fmt.Sprintf("invalid argument to Shuffle: %d", n))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rng.Shuffle(n, swap)
}
