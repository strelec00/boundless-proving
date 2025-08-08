# export_weights_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Your SimpleMLP model definition (must match training exactly)
class OptimizedSimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)
        
        # Kaiming initialization for ReLU (matches training)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create output directory
os.makedirs('./weights', exist_ok=True)

# Load the trained model
model = OptimizedSimpleMLP()

# Try to load the best model (prioritize student model from your training)
model_paths = [
    './models/best_student_model.pth',
    './models/best_binary_model.pth',
    './models/best_teacher_model.pth',  # Fallback
]

model_loaded = False
for path in model_paths:
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"✅ Loaded model from: {path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            continue

if not model_loaded:
    print("❌ No model file found! Make sure to run training first.")
    exit(1)

model.eval()

# SCALE matches your training code exactly
SCALE = 16384

def to_fixed(tensor, scale=SCALE):
    """Convert to fixed-point with proper rounding (matches training)"""
    scaled = tensor.detach().cpu().numpy() * scale
    rounded = np.round(scaled)
    return np.clip(rounded, -2147483648, 2147483647).astype(np.int32)

# Fold batch normalization into fc1 weights (exactly like in training)
bn_mean = model.bn1.running_mean.detach().cpu()
bn_var = model.bn1.running_var.detach().cpu()
bn_weight = model.bn1.weight.detach().cpu()
bn_bias = model.bn1.bias.detach().cpu()
eps = model.bn1.eps

# Calculate folded weights
std = torch.sqrt(bn_var + eps)
fc1_weight = model.fc1.weight.detach().cpu()
fc1_bias = model.fc1.bias.detach().cpu()

# Fold BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
fc1_weight_folded = fc1_weight * (bn_weight / std).unsqueeze(1)
fc1_bias_folded = bn_weight * (fc1_bias - bn_mean) / std + bn_bias

# Prepare weights (transpose for row-major access in Rust)
W1 = fc1_weight_folded.T  # [784, 64]
B1 = fc1_bias_folded      # [64]
W2 = model.fc2.weight.T   # [64, 10]
B2 = model.fc2.bias       # [10]

# Convert to fixed-point
W1_fixed = to_fixed(W1)
B1_fixed = to_fixed(B1)
W2_fixed = to_fixed(W2)
B2_fixed = to_fixed(B2)

# Generate Rust include files
def write_rust_array_2d(filename, array, name):
    """Write 2D array as Rust const"""
    with open(filename, 'w') as f:
        rows, cols = array.shape
        f.write(f"pub const {name}: [[i32; {cols}]; {rows}] = [\n")
        for i in range(rows):
            f.write("    [")
            for j in range(cols):
                f.write(f"{array[i, j]}")
                if j < cols - 1:
                    f.write(", ")
            f.write("]")
            if i < rows - 1:
                f.write(",")
            f.write("\n")
        f.write("];\n")

def write_rust_array_1d(filename, array, name):
    """Write 1D array as Rust const"""
    with open(filename, 'w') as f:
        size = len(array)
        f.write(f"pub const {name}: [i32; {size}] = [\n    ")
        for i, val in enumerate(array):
            f.write(f"{val}")
            if i < size - 1:
                f.write(", ")
            if (i + 1) % 10 == 0 and i < size - 1:
                f.write("\n    ")
        f.write("\n];\n")

# Write the include files
write_rust_array_2d('./weights/W1.incl.rs', W1_fixed, 'W1')
write_rust_array_1d('./weights/B1.incl.rs', B1_fixed, 'B1')
write_rust_array_2d('./weights/W2.incl.rs', W2_fixed, 'W2')
write_rust_array_1d('./weights/B2.incl.rs', B2_fixed, 'B2')

# Also save as numpy files for compatibility with your training script
np.save("./weights/W1.npy", W1_fixed)
np.save("./weights/B1.npy", B1_fixed)
np.save("./weights/W2.npy", W2_fixed)
np.save("./weights/B2.npy", B2_fixed)
np.save("./weights/scale.npy", np.array([SCALE]))

print(f"\n✅ Weights exported with SCALE={SCALE}")
print(f"📊 Model statistics:")
print(f"   - W1: [{W1_fixed.shape[0]}, {W1_fixed.shape[1]}] -> fc1 weights (with folded BN)")
print(f"   - B1: [{len(B1_fixed)}] -> fc1 bias (with folded BN)")
print(f"   - W2: [{W2_fixed.shape[0]}, {W2_fixed.shape[1]}] -> fc2 weights")
print(f"   - B2: [{len(B2_fixed)}] -> fc2 bias")

print(f"\n📁 Files created:")
print(f"   - Rust includes: ./weights/*.incl.rs")
print(f"   - NumPy arrays: ./weights/*.npy")
print(f"\n💡 Copy the .incl.rs files to your guest/src/weights/ directory")

# Validate the conversion by testing inference
print(f"\n🧪 Testing converted weights...")

# Test with a simple example (zeros input)
test_input = torch.zeros(1, 784)
with torch.no_grad():
    original_output = model(test_input)
    
# Manual forward pass with fixed-point weights
def fixed_to_float(fixed_weights, scale=SCALE):
    return fixed_weights.astype(np.float32) / scale

# Convert back to float for testing
W1_float = fixed_to_float(W1_fixed)
B1_float = fixed_to_float(B1_fixed)
W2_float = fixed_to_float(W2_fixed)
B2_float = fixed_to_float(B2_fixed)

# Manual inference
x = test_input.numpy().reshape(-1)
h1 = np.maximum(0, np.dot(x, W1_float) + B1_float)  # ReLU
output_manual = np.dot(h1, W2_float) + B2_float

print(f"Original output (first 3): {original_output[0][:3].numpy()}")
print(f"Manual output (first 3):   {output_manual[:3]}")
print(f"Max difference: {np.max(np.abs(original_output[0].numpy() - output_manual)):.6f}")

if np.max(np.abs(original_output[0].numpy() - output_manual)) < 0.01:
    print("✅ Conversion validation passed!")
else:
    print("⚠️  Large difference detected - check conversion logic")

print("\n🎯 Export complete! Your weights are ready for inference.")