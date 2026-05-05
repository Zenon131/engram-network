# Enhanced BDH Model with Receptive Fields, Grid Cells, and Successor Representations

## 1. Summary

This document describes an enhanced Brain-inspired Differentiable Hebbian (BDH) model that incorporates:

- Receptive fields with local connectivity via 2D convolution
- 2D grid layout for neurons with place cell coordinates
- Grid cell sinusoidal features for spatial coding
- State representation derived from neuron activations
- Successor Representation (SR) learning over internal state space
- Integration of SR features into the model's next-token prediction pipeline
- Compatibility with BDH-style fast-weight/Hebbian updates

## 2. Technical Breakdown

### 2.1 Receptive Fields and Local Connectivity

The model uses 2D convolution to implement local receptive fields for neurons arranged on a grid:

```
x_processed = receptive_field(x_reshaped)
```

Where:
- `x` is neuron activations of shape (B, n)
- `x_reshaped` is of shape (B, 1, grid_h, grid_w)
- `receptive_field` is a Conv2D layer with local connectivity weights

### 2.2 Grid/Place Coding

Neurons are arranged on a 2D grid with:
- Place coordinates: (i, j) for each neuron
- Grid cell features: Sinusoidal functions at multiple scales and orientations

Grid features for neuron at position (i,j):
```
feat[scale, orient] = [sin(2πx_proj/(scale*grid_w)), cos(2πy_proj/(scale*grid_h))]
```

Where:
- `x_proj = i*cos(θ) + j*sin(θ)`
- `y_proj = -i*sin(θ) + j*cos(θ)`
- `θ = orient * π / n_orientations`

### 2.3 State Distribution Extraction

State features are extracted from neuron activations using grid features as a basis:
```
state_features = y @ grid_features
```

Where:
- `y` is neuron activations of shape (B, n)
- `grid_features` is of shape (n, feature_dim)
- `state_features` is of shape (B, feature_dim)

### 2.4 Successor Representation Learning

SR is learned via a TD-like rule:
```
ψ_t = Wϕ(s_t)
W ← W + α(ϕ(s_t) + γWϕ(s_{t+1}) - ψ_t)ϕ(s_t)^T
```

Where:
- `ϕ(s_t)` are state features at time t
- `ψ_t` are successor representations at time t
- `W` is the SR matrix
- `α` is the learning rate
- `γ` is the discount factor

### 2.5 SR Feature Integration

SR features are concatenated with the final state representation for next-token prediction:
```
combined_features = cat([v_final, sr_features], dim=-1)
logits = readout(combined_features)
```

## 3. Implementation Steps

### 3.1 Neural Sheet Layout

1. Create a `NeuronGrid` class with:
   - Grid size (grid_h, grid_w)
   - Place coordinates for each neuron
   - Grid cell features (sinusoidal)

2. Ensure n_neurons = grid_h * grid_w

### 3.2 Receptive Field Conv Kernels

1. Implement `ReceptiveFieldConv2D` class:
   - Use Conv2D with kernel_size (3x3 or larger)
   - Initialize weights to emphasize local connections
   - Reshape 1D neuron activations to 2D grid for convolution
   - Reshape back to 1D after convolution

### 3.3 Grid/Place Coding Generation

1. In `NeuronGrid`:
   - Generate place coordinates in `_create_place_coords()`
   - Generate grid features in `_create_grid_features()`
   - Support multiple scales and orientations

### 3.4 State Distribution Extraction

1. In enhanced BDH layer:
   - Implement `extract_state_features(y)` method
   - Use grid features as basis: `state_features = y @ grid_features`

### 3.5 SR Update Rule

1. Implement `SuccessorRepresentation` class:
   - Store SR matrix W
   - Implement `compute_sr(state_features)` method
   - Implement `update_sr(current_features, next_features)` method with TD update

### 3.6 Inject SR Features into Logits/Output Head

1. Modify BDH model readout:
   - Concatenate SR features with final state representation
   - Update readout layer input dimension (d * 2)
   - Compute logits from combined features

### 3.7 Training, Loss, and Evaluation

1. No changes to training loss (still cross-entropy on tokens)
2. SR updates happen internally during forward pass
3. Evaluation can access state_features and sr_features for analysis

## 4. Optional Enhancements

### 4.1 Multiscale Receptive Fields

- Use dilated convolutions with different dilation rates
- Combine multiple conv layers with different receptive fields

### 4.2 Dilated Convs

- Replace standard conv with dilated convolutions
- Allows larger receptive fields without increasing parameters

### 4.3 Additional Hippocampal Modules

- Add replay-based consolidation mechanism
- Implement multi-head SR representations for different timescales

### 4.4 Replay-based Consolidation

- Store important state transitions
- Re-run SR updates on stored transitions

### 4.5 Multi-head SR Representations

- Learn multiple SR matrices for different γ values
- Allows representing different timescales of future predictions

## 5. Example Code

### 5.1 2D Neuron Grid Creation

```python
grid = NeuronGrid(grid_size=(16, 16), n_neurons=256)
```

### 5.2 Local Receptive-Field Conv Operations

```python
conv_layer = ReceptiveFieldConv2D(n_neurons=256, grid_size=(16, 16), kernel_size=3)
x_processed = conv_layer(x)
```

### 5.3 Grid Cell Feature Generation

```python
# In NeuronGrid.__init__
self.grid_features = self._create_grid_features()
```

### 5.4 SR TD Update Rule

```python
# In SuccessorRepresentation.update_sr
td_error = current_features + self.gamma * next_sr - current_sr
self.W += self.learning_rate * torch.einsum("bd,bD->dD", td_error, current_features)
```

### 5.5 Combined BDH + SR Step

```python
# In EnhancedBDHLayer.forward
x, y, rho, state_features, sr_features = layer(x, y, rho, state_features)
```

### 5.6 Readout with SR Concatenation

```python
# In EnhancedBDH_GPU.forward
combined_features = torch.cat([v_final, sr_features], dim=-1)
logits = self.readout(combined_features)