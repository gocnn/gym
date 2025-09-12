package gym

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"sync"
)

// EnvCreator is a function type that creates an environment
type EnvCreator[ObsType, ActType any] func(config any) (Env[ObsType, ActType], error)

// EnvSpec holds the specification for an environment
type EnvSpec[ObsType, ActType any] struct {
	ID                string
	EntryPoint        EnvCreator[ObsType, ActType]
	RewardThreshold   *float64
	Nondeterministic  bool
	MaxEpisodeSteps   *int
	OrderEnforce      bool
	DisableEnvChecker bool
	Kwargs            map[string]any

	// Parsed components
	Namespace *string
	Name      string
	Version   *int
}

// Registry holds all registered environments
type Registry struct {
	mu    sync.RWMutex
	specs map[string]any // map[string]*EnvSpec[ObsType, ActType]
}

var (
	// Global registry instance
	registry = &Registry{
		specs: make(map[string]any),
	}

	// Environment ID regex pattern
	envIDRegex = regexp.MustCompile(`^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$`)
)

// ParseEnvID parses environment ID string format
func ParseEnvID(envID string) (namespace *string, name string, version *int, err error) {
	matches := envIDRegex.FindStringSubmatch(envID)
	if matches == nil {
		return nil, "", nil, fmt.Errorf("malformed environment ID: %s", envID)
	}

	// Extract named groups
	result := make(map[string]string)
	for i, name := range envIDRegex.SubexpNames() {
		if i != 0 && name != "" && i < len(matches) {
			result[name] = matches[i]
		}
	}

	// Parse namespace
	if ns, ok := result["namespace"]; ok && ns != "" {
		namespace = &ns
	}

	// Parse name
	name = result["name"]

	// Parse version
	if v, ok := result["version"]; ok && v != "" {
		if ver, err := strconv.Atoi(v); err == nil {
			version = &ver
		}
	}

	return namespace, name, version, nil
}

// GetEnvID constructs environment ID from components
func GetEnvID(namespace *string, name string, version *int) string {
	fullName := name
	if namespace != nil {
		fullName = fmt.Sprintf("%s/%s", *namespace, name)
	}
	if version != nil {
		fullName = fmt.Sprintf("%s-v%d", fullName, *version)
	}
	return fullName
}

// Register registers an environment in the global registry
func Register[ObsType, ActType any](
	id string,
	entryPoint EnvCreator[ObsType, ActType],
	options ...func(*EnvSpec[ObsType, ActType]),
) error {
	namespace, name, version, err := ParseEnvID(id)
	if err != nil {
		return err
	}

	spec := &EnvSpec[ObsType, ActType]{
		ID:                GetEnvID(namespace, name, version),
		EntryPoint:        entryPoint,
		Namespace:         namespace,
		Name:              name,
		Version:           version,
		OrderEnforce:      true,
		DisableEnvChecker: false,
		Kwargs:            make(map[string]interface{}),
	}

	// Apply options
	for _, option := range options {
		option(spec)
	}

	registry.mu.Lock()
	defer registry.mu.Unlock()

	registry.specs[spec.ID] = spec
	return nil
}

// Option functions for Register
func WithRewardThreshold[ObsType, ActType any](threshold float64) func(*EnvSpec[ObsType, ActType]) {
	return func(spec *EnvSpec[ObsType, ActType]) {
		spec.RewardThreshold = &threshold
	}
}

func WithMaxEpisodeSteps[ObsType, ActType any](steps int) func(*EnvSpec[ObsType, ActType]) {
	return func(spec *EnvSpec[ObsType, ActType]) {
		spec.MaxEpisodeSteps = &steps
	}
}

func WithNondeterministic[ObsType, ActType any](nondeterministic bool) func(*EnvSpec[ObsType, ActType]) {
	return func(spec *EnvSpec[ObsType, ActType]) {
		spec.Nondeterministic = nondeterministic
	}
}

func WithKwargs[ObsType, ActType any](kwargs map[string]interface{}) func(*EnvSpec[ObsType, ActType]) {
	return func(spec *EnvSpec[ObsType, ActType]) {
		for k, v := range kwargs {
			spec.Kwargs[k] = v
		}
	}
}

// Make creates an environment from the registry
func Make[ObsType, ActType any](id string, kwargs ...map[string]interface{}) (Env[ObsType, ActType], error) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	specInterface, exists := registry.specs[id]
	if !exists {
		return nil, fmt.Errorf("no registered environment with id: %s", id)
	}

	// Type assertion to get the correct spec type
	spec, ok := specInterface.(*EnvSpec[ObsType, ActType])
	if !ok {
		return nil, fmt.Errorf("environment %s has incompatible types", id)
	}

	// Merge kwargs
	config := make(map[string]interface{})
	for k, v := range spec.Kwargs {
		config[k] = v
	}
	for _, kw := range kwargs {
		for k, v := range kw {
			config[k] = v
		}
	}

	// Create environment
	env, err := spec.EntryPoint(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create environment %s: %w", id, err)
	}

	return env, nil
}

// MakeAny creates an environment without type constraints (returns interface{})
func MakeAny(id string, kwargs ...map[string]interface{}) (interface{}, error) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	specInterface, exists := registry.specs[id]
	if !exists {
		return nil, fmt.Errorf("no registered environment with id: %s", id)
	}

	// Use reflection to call the entry point
	specValue := reflect.ValueOf(specInterface)
	entryPointField := specValue.Elem().FieldByName("EntryPoint")

	if !entryPointField.IsValid() {
		return nil, fmt.Errorf("invalid entry point for environment %s", id)
	}

	// Merge kwargs
	config := make(map[string]interface{})

	// Get spec kwargs using reflection
	kwargsField := specValue.Elem().FieldByName("Kwargs")
	if kwargsField.IsValid() && kwargsField.Kind() == reflect.Map {
		for _, key := range kwargsField.MapKeys() {
			config[key.String()] = kwargsField.MapIndex(key).Interface()
		}
	}

	for _, kw := range kwargs {
		for k, v := range kw {
			config[k] = v
		}
	}

	// Call entry point function
	results := entryPointField.Call([]reflect.Value{reflect.ValueOf(config)})
	if len(results) != 2 {
		return nil, fmt.Errorf("invalid entry point signature for environment %s", id)
	}

	env := results[0].Interface()
	errInterface := results[1].Interface()

	if errInterface != nil {
		if err, ok := errInterface.(error); ok {
			return nil, fmt.Errorf("failed to create environment %s: %w", id, err)
		}
	}

	return env, nil
}

// Spec returns the environment specification for the given ID
func Spec(id string) (interface{}, error) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	spec, exists := registry.specs[id]
	if !exists {
		return nil, fmt.Errorf("no registered environment with id: %s", id)
	}

	return spec, nil
}

// ListRegistered returns all registered environment IDs
func ListRegistered() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	ids := make([]string, 0, len(registry.specs))
	for id := range registry.specs {
		ids = append(ids, id)
	}
	return ids
}

// PrintRegistry prints all registered environments
func PrintRegistry() {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	fmt.Println("Registered Environments:")
	fmt.Println("========================")

	for id := range registry.specs {
		fmt.Printf("- %s\n", id)
	}
}
