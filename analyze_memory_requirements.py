import numpy as np

def calculate_memory_requirements(n_neurons, d_model=256, n_layers=6, batch_size=32, seq_length=512, dtype_bytes=4):
    """Calculate memory requirements for BDH model"""
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

# Current configurations
configs = [
    ('tiny', 1024),
    ('small', 8192),
    ('base', 32768),
    ('large', 65536),
    ('300M', 300000000)
]

print('Memory Requirements Analysis:')
print('=' * 80)
print('Config        Neurons           Params   Param Mem (MB)   Activation Mem (MB)   Total Mem (MB)')
print('-' * 80)

for name, neurons in configs:
    params, param_mem, act_mem, total_mem = calculate_memory_requirements(neurons)
    print(f'{name:<10} {neurons:>12,} {params:>15,} {param_mem:>15.1f} {act_mem:>20.1f} {total_mem:>15.1f}')

print('\nMemory Constraints:')
print(f'16GB RAM = {16 * 1024:.0f} MB')
print('Note: This analysis assumes float32 (4 bytes per parameter)')
print('Real usage will be higher due to optimizer states, gradients, and system overhead')

# Additional analysis for different batch sizes
print('\n\nMemory Requirements for 300M neurons with different batch sizes:')
print('=' * 60)
print('Batch Size   Activation Mem (MB)   Total Mem (MB)')
print('-' * 60)

batch_sizes = [1, 2, 4, 8, 16, 32]
for batch_size in batch_sizes:
    _, param_mem, act_mem, total_mem = calculate_memory_requirements(300000000, batch_size=batch_size)
    print(f'{batch_size:>10} {act_mem:>20.1f} {total_mem:>15.1f}')

# Analysis with mixed precision (float16)
print('\n\nMemory Requirements with float16 (2 bytes per parameter):')
print('=' * 60)
print('Config        Neurons   Total Mem (MB) float32   Total Mem (MB) float16')
print('-' * 60)

for name, neurons in configs:
    _, _, _, total_mem_32 = calculate_memory_requirements(neurons, dtype_bytes=4)
    _, _, _, total_mem_16 = calculate_memory_requirements(neurons, dtype_bytes=2)
    print(f'{name:<10} {neurons:>12,} {total_mem_32:>20.1f} {total_mem_16:>20.1f}')