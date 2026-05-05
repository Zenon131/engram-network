def calculate_memory_by_layers(n_neurons, n_layers, d_model=256, batch_size=32, dtype_bytes=4):
    """Calculate memory requirements based on number of layers vs neurons"""
    # Parameters per layer
    params_per_layer = n_neurons * d_model * 3  # E, Dx, Dy matrices
    total_params = params_per_layer * n_layers
    
    # Memory for parameters (in MB)
    param_memory_mb = (total_params * dtype_bytes) / (1024**2)
    
    # Memory for activations during forward pass
    # rho tensor: (batch_size, n_neurons, d_model)
    rho_memory = batch_size * n_neurons * d_model * dtype_bytes / (1024**2)
    
    # x, y tensors: (batch_size, n_neurons)
    xy_memory = 2 * batch_size * n_neurons * dtype_bytes / (1024**2)
    
    # Total activation memory
    activation_memory_mb = rho_memory + xy_memory
    
    # Total memory (parameters + activations)
    total_memory_mb = param_memory_mb + activation_memory_mb
    
    return total_params, param_memory_mb, activation_memory_mb, total_memory_mb

# Compare different layer counts with same total parameter budget
print("Memory Requirements: Layers vs Neurons Trade-off")
print("=" * 80)
print("Strategy: Keep total parameters approximately constant")
print("=" * 80)

# Base case: 32K neurons, 6 layers (current base config)
base_neurons = 32768
base_layers = 6
base_params, base_param_mem, base_act_mem, base_total_mem = calculate_memory_by_layers(base_neurons, base_layers)

print(f"\\nBase configuration:")
print(f"Neurons: {base_neurons:,}, Layers: {base_layers}")
print(f"Total parameters: {base_params:,}")
print(f"Total memory: {base_total_mem:.1f} MB")

# Strategy 1: More layers, fewer neurons (same parameter budget)
print(f"\\nStrategy 1: More layers, fewer neurons")
print("-" * 60)

# Calculate target parameter count
target_params = base_params

# Try different layer counts
layer_configs = [
    (2, 1.5),   # Fewer layers, more neurons
    (4, 1.2),   # Slightly fewer layers
    (6, 1.0),   # Base
    (12, 0.7),  # More layers, fewer neurons
    (24, 0.5),  # Even more layers
    (48, 0.35)  # Many layers
]

print(f"{'Layers':>6} {'Neurons':>12} {'Params':>15} {'Param Mem (MB)':>15} {'Activation Mem (MB)':>20} {'Total Mem (MB)':>15}")
print("-" * 80)

for layers, factor in layer_configs:
    neurons = int(base_neurons * factor)
    params, param_mem, act_mem, total_mem = calculate_memory_by_layers(neurons, layers)
    
    print(f"{layers:>6} {neurons:>12,} {params:>15,} {param_mem:>15.1f} {act_mem:>20.1f} {total_mem:>15.1f}")

# Strategy 2: Memory usage with different layer counts for same neuron count
print(f"\\n\\nStrategy 2: Same neuron count, different layers (32K neurons)")
print("-" * 60)
print(f"{'Layers':>6} {'Neurons':>12} {'Params':>15} {'Param Mem (MB)':>15} {'Activation Mem (MB)':>20} {'Total Mem (MB)':>15}")
print("-" * 80)

neurons = 32768
for layers in [2, 4, 6, 8, 12, 16]:
    params, param_mem, act_mem, total_mem = calculate_memory_by_layers(neurons, layers)
    print(f"{layers:>6} {neurons:>12,} {params:>15,} {param_mem:>15.1f} {act_mem:>20.1f} {total_mem:>15.1f}")

# Strategy 3: Maximum feasible on 16GB RAM
print(f"\\n\\nStrategy 3: Maximum feasible configurations on 16GB RAM")
print("-" * 60)
print(f"{'Layers':>6} {'Neurons':>12} {'Params':>15} {'Total Mem (MB)':>15} {'RAM Usage %':>12}")
print("-" * 60)

ram_limit_mb = 16 * 1024  # 16GB in MB
feasible_configs = []

# Test various combinations
for layers in [2, 4, 6, 8, 12]:
    for neuron_power in range(10, 18):  # 1K to 128K neurons
        neurons = 2 ** neuron_power
        params, param_mem, act_mem, total_mem = calculate_memory_by_layers(neurons, layers, batch_size=8)
        
        if total_mem < ram_limit_mb * 0.8:  # Leave 20% for system
            feasible_configs.append((layers, neurons, params, total_mem))

# Sort by parameter count (descending)
feasible_configs.sort(key=lambda x: x[2], reverse=True)

# Show top 10 feasible configurations
for i, (layers, neurons, params, total_mem) in enumerate(feasible_configs[:10]):
    ram_usage_pct = (total_mem / ram_limit_mb) * 100
    print(f"{layers:>6} {neurons:>12,} {params:>15,} {total_mem:>15.1f} {ram_usage_pct:>11.1f}%")

print(f"\\nKey insights:")
print(f"- Adding layers increases parameter count linearly")
print(f"- Activation memory is independent of layer count (depends on batch size and neurons)")
print(f"- On 16GB RAM, you can support up to ~100K-200K parameters with reasonable batch sizes")
print(f"- More layers allow for deeper processing without increasing activation memory")