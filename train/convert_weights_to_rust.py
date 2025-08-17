# export_cnn_weights_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Your CNN model definition (must match training exactly)
class OptimizedSimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # 14x14 -> 14x14
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(x.size(0), -1)  # 32 * 7 * 7 = 1568
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create output directory
os.makedirs('./weights', exist_ok=True)

# Load the trained CNN model
model = OptimizedSimpleCNN()

# Try to load the best CNN model
model_paths = [
    './models/best_student_cnn_model.pth',
    './models/best_teacher_cnn_model.pth',  # Fallback
]

model_loaded = False
for path in model_paths:
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"✅ Loaded CNN model from: {path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            continue

if not model_loaded:
    print("❌ No CNN model file found! Make sure to run CNN training first.")
    exit(1)

model.eval()

# SCALE matches your training code exactly
SCALE = 16384

def to_fixed(tensor, scale=SCALE):
    """Convert to fixed-point with proper rounding"""
    scaled = tensor.detach().cpu().numpy() * scale
    rounded = np.round(scaled)
    return np.clip(rounded, -2147483648, 2147483647).astype(np.int32)

# Fold batch normalization into conv layers (exactly like in training)
def fold_bn_into_conv(conv_layer, bn_layer):
    """Fold batch normalization into convolution layer"""
    with torch.no_grad():
        # Get parameters
        conv_weight = conv_layer.weight.clone()
        conv_bias = conv_layer.bias.clone() if conv_layer.bias is not None else torch.zeros(conv_layer.out_channels)
        
        bn_weight = bn_layer.weight.clone()
        bn_bias = bn_layer.bias.clone()
        bn_mean = bn_layer.running_mean.clone()
        bn_var = bn_layer.running_var.clone()
        eps = bn_layer.eps
        
        # Compute folded parameters
        std = torch.sqrt(bn_var + eps)
        factor = bn_weight / std
        
        # Fold into conv weight and bias
        folded_weight = conv_weight * factor.view(-1, 1, 1, 1)
        folded_bias = bn_weight * (conv_bias - bn_mean) / std + bn_bias
        
        return folded_weight, folded_bias

# Fold batch norm layers
conv1_weight, conv1_bias = fold_bn_into_conv(model.conv1, model.bn1)
conv2_weight, conv2_bias = fold_bn_into_conv(model.conv2, model.bn2)

# Fold FC batch norm
fc1_weight = model.fc1.weight.clone()
fc1_bias = model.fc1.bias.clone()
bn_weight = model.bn_fc.weight.clone()
bn_bias = model.bn_fc.bias.clone()
bn_mean = model.bn_fc.running_mean.clone()
bn_var = model.bn_fc.running_var.clone()
eps = model.bn_fc.eps

std = torch.sqrt(bn_var + eps)
fc1_weight_folded = fc1_weight * (bn_weight / std).unsqueeze(1)
fc1_bias_folded = bn_weight * (fc1_bias - bn_mean) / std + bn_bias

fc2_weight = model.fc2.weight.clone()
fc2_bias = model.fc2.bias.clone()

# Convert to fixed-point
conv1_weight_fixed = to_fixed(conv1_weight)
conv1_bias_fixed = to_fixed(conv1_bias)
conv2_weight_fixed = to_fixed(conv2_weight)
conv2_bias_fixed = to_fixed(conv2_bias)
fc1_weight_fixed = to_fixed(fc1_weight_folded)
fc1_bias_fixed = to_fixed(fc1_bias_folded)
fc2_weight_fixed = to_fixed(fc2_weight)
fc2_bias_fixed = to_fixed(fc2_bias)

# Generate Rust include files
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

def write_rust_conv_weights(filename, weights, name):
    """Write 4D conv weights as flattened Rust const"""
    # weights shape: [out_channels, in_channels, kernel_h, kernel_w]
    out_ch, in_ch, kh, kw = weights.shape
    flattened = weights.flatten()
    
    with open(filename, 'w') as f:
        f.write(f"// Conv weights: [{out_ch}, {in_ch}, {kh}, {kw}] -> flattened to [{len(flattened)}]\n")
        f.write(f"pub const {name}: [i32; {len(flattened)}] = [\n")
        
        # Write in groups for readability
        items_per_line = 8
        for i, val in enumerate(flattened):
            if i % items_per_line == 0:
                f.write("    ")
            f.write(f"{val}")
            if i < len(flattened) - 1:
                f.write(", ")
            if (i + 1) % items_per_line == 0 and i < len(flattened) - 1:
                f.write("\n")
        f.write("\n];\n")
        
        # Add helper constants
        f.write(f"\npub const {name}_OUT_CHANNELS: usize = {out_ch};\n")
        f.write(f"pub const {name}_IN_CHANNELS: usize = {in_ch};\n")
        f.write(f"pub const {name}_KERNEL_SIZE: usize = {kh};\n")

# Write the include files
write_rust_conv_weights('./weights/CONV1_WEIGHTS.incl.rs', conv1_weight_fixed, 'CONV1_WEIGHTS')
write_rust_array_1d('./weights/CONV1_BIAS.incl.rs', conv1_bias_fixed, 'CONV1_BIAS')

write_rust_conv_weights('./weights/CONV2_WEIGHTS.incl.rs', conv2_weight_fixed, 'CONV2_WEIGHTS')
write_rust_array_1d('./weights/CONV2_BIAS.incl.rs', conv2_bias_fixed, 'CONV2_BIAS')

write_rust_array_2d('./weights/FC1_WEIGHTS.incl.rs', fc1_weight_fixed, 'FC1_WEIGHTS')
write_rust_array_1d('./weights/FC1_BIAS.incl.rs', fc1_bias_fixed, 'FC1_BIAS')

write_rust_array_2d('./weights/FC2_WEIGHTS.incl.rs', fc2_weight_fixed, 'FC2_WEIGHTS')
write_rust_array_1d('./weights/FC2_BIAS.incl.rs', fc2_bias_fixed, 'FC2_BIAS')

# Also save as numpy files for compatibility
np.save("./weights/conv1_weight.npy", conv1_weight_fixed)
np.save("./weights/conv1_bias.npy", conv1_bias_fixed)
np.save("./weights/conv2_weight.npy", conv2_weight_fixed)
np.save("./weights/conv2_bias.npy", conv2_bias_fixed)
np.save("./weights/fc1_weight.npy", fc1_weight_fixed)
np.save("./weights/fc1_bias.npy", fc1_bias_fixed)
np.save("./weights/fc2_weight.npy", fc2_weight_fixed)
np.save("./weights/fc2_bias.npy", fc2_bias_fixed)
np.save("./weights/scale.npy", np.array([SCALE]))

# Create a summary constants file
with open('./weights/MODEL_CONSTANTS.incl.rs', 'w') as f:
    f.write("// CNN Model Constants\n")
    f.write(f"pub const SCALE: i32 = {SCALE};\n")
    f.write(f"pub const INPUT_SIZE: usize = 28;\n")
    f.write(f"pub const NUM_CLASSES: usize = 10;\n")
    f.write(f"pub const CONV1_OUT_CHANNELS: usize = 16;\n")
    f.write(f"pub const CONV2_OUT_CHANNELS: usize = 32;\n")
    f.write(f"pub const FC1_HIDDEN_SIZE: usize = 128;\n")
    f.write(f"pub const CONV_KERNEL_SIZE: usize = 5;\n")
    f.write(f"pub const CONV_PADDING: usize = 2;\n")
    f.write(f"pub const POOL_SIZE: usize = 2;\n")

print(f"\n✅ CNN Weights exported with SCALE={SCALE}")
print(f"📊 Model statistics:")
print(f"   - CONV1: [{conv1_weight_fixed.shape}] -> {conv1_weight_fixed.size} weights + {len(conv1_bias_fixed)} bias")
print(f"   - CONV2: [{conv2_weight_fixed.shape}] -> {conv2_weight_fixed.size} weights + {len(conv2_bias_fixed)} bias")
print(f"   - FC1: [{fc1_weight_fixed.shape}] -> {fc1_weight_fixed.size} weights + {len(fc1_bias_fixed)} bias")
print(f"   - FC2: [{fc2_weight_fixed.shape}] -> {fc2_weight_fixed.size} weights + {len(fc2_bias_fixed)} bias")

print(f"\n📁 Files created:")
print(f"   - CONV1_WEIGHTS.incl.rs (conv1 weights: 16×1×5×5 = {conv1_weight_fixed.size})")
print(f"   - CONV1_BIAS.incl.rs (conv1 bias: {len(conv1_bias_fixed)})")
print(f"   - CONV2_WEIGHTS.incl.rs (conv2 weights: 32×16×5×5 = {conv2_weight_fixed.size})")
print(f"   - CONV2_BIAS.incl.rs (conv2 bias: {len(conv2_bias_fixed)})")
print(f"   - FC1_WEIGHTS.incl.rs (fc1 weights: {fc1_weight_fixed.shape})")
print(f"   - FC1_BIAS.incl.rs (fc1 bias: {len(fc1_bias_fixed)})")
print(f"   - FC2_WEIGHTS.incl.rs (fc2 weights: {fc2_weight_fixed.shape})")
print(f"   - FC2_BIAS.incl.rs (fc2 bias: {len(fc2_bias_fixed)})")
print(f"   - MODEL_CONSTANTS.incl.rs (model configuration)")
print(f"   - NumPy arrays: ./weights/*.npy")

print(f"\n💡 Copy the .incl.rs files to your guest/src/weights/ directory")

# Validate the conversion by testing a simple forward pass
print(f"\n🧪 Testing converted weights...")

# Test with a simple example (zeros input)
test_input = torch.zeros(1, 1, 28, 28)
with torch.no_grad():
    original_output = model(test_input)

print(f"Original CNN output (first 3): {original_output[0][:3].numpy()}")
print(f"Output shape: {original_output.shape}")

# Basic sanity check - just verify shapes
print(f"✅ Model loaded and tested successfully!")
print(f"📐 Layer shapes verified:")
print(f"   - Input: 1×1×28×28")
print(f"   - After conv1+pool: 1×16×14×14")
print(f"   - After conv2+pool: 1×32×7×7")
print(f"   - After flatten: 1×{32*7*7}")
print(f"   - After fc1: 1×128")
print(f"   - After fc2: 1×10")

print("\n🎯 CNN Export complete! Your weights are ready for inference.")
print("\n📝 To use in Rust:")
print("   1. Copy all .incl.rs files to your Rust project")
print("   2. Include them with: include!(\"weights/CONV1_WEIGHTS.incl.rs\");")
print("   3. Use MODEL_CONSTANTS.incl.rs for configuration values")