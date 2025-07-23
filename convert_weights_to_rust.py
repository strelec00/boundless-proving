#!/usr/bin/env python3
"""
Script za konverziju težina iz numpy format u Rust include fajlove
"""

import numpy as np
import os

def convert_weights_to_rust():
    """Konvertuje .npy fajlove u Rust include fajlove"""
    
    weights_dir = "./weights"
    
    # Učitaj numpy fajlove
    W1 = np.load(os.path.join(weights_dir, "W1.npy"))
    B1 = np.load(os.path.join(weights_dir, "B1.npy"))
    W2 = np.load(os.path.join(weights_dir, "W2.npy"))
    B2 = np.load(os.path.join(weights_dir, "B2.npy"))
    
    print(f"W1 shape: {W1.shape}")
    print(f"B1 shape: {B1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"B2 shape: {B2.shape}")
    
    # Kreiraj Rust include fajlove
    
    # W1: 784x64 matrix
    with open(os.path.join(weights_dir, "W1.incl.rs"), "w") as f:
        f.write("// Auto-generated weights for layer 1\n")
        f.write("pub const W1: [[i32; 64]; 784] = [\n")
        for i in range(W1.shape[0]):
            f.write("    [")
            for j in range(W1.shape[1]):
                f.write(f"{W1[i, j]}")
                if j < W1.shape[1] - 1:
                    f.write(", ")
            f.write("]")
            if i < W1.shape[0] - 1:
                f.write(",")
            f.write("\n")
        f.write("];\n")
    
    # B1: 64 element array
    with open(os.path.join(weights_dir, "B1.incl.rs"), "w") as f:
        f.write("// Auto-generated bias for layer 1\n")
        f.write("pub const B1: [i32; 64] = [")
        for i in range(B1.shape[0]):
            f.write(f"{B1[i]}")
            if i < B1.shape[0] - 1:
                f.write(", ")
        f.write("];\n")
    
    # W2: 64x10 matrix
    with open(os.path.join(weights_dir, "W2.incl.rs"), "w") as f:
        f.write("// Auto-generated weights for layer 2\n")
        f.write("pub const W2: [[i32; 10]; 64] = [\n")
        for i in range(W2.shape[0]):
            f.write("    [")
            for j in range(W2.shape[1]):
                f.write(f"{W2[i, j]}")
                if j < W2.shape[1] - 1:
                    f.write(", ")
            f.write("]")
            if i < W2.shape[0] - 1:
                f.write(",")
            f.write("\n")
        f.write("];\n")
    
    # B2: 10 element array
    with open(os.path.join(weights_dir, "B2.incl.rs"), "w") as f:
        f.write("// Auto-generated bias for layer 2\n")
        f.write("pub const B2: [i32; 10] = [")
        for i in range(B2.shape[0]):
            f.write(f"{B2[i]}")
            if i < B2.shape[0] - 1:
                f.write(", ")
        f.write("];\n")
    
    print("Rust include fajlovi kreirani u ./weights/ direktorijumu:")
    print("- W12.incl.rs")
    print("- B12.incl.rs")
    print("- W22.incl.rs")
    print("- B22.incl.rs")
    
    # Kreiranje guest/src/weights direktorijuma ako ne postoji
    guest_weights_dir = "guests/mnist_prediction/src/weights"
    os.makedirs(guest_weights_dir, exist_ok=True)
    
    # Kopiraj fajlove u guest direktorijum
    import shutil
    for filename in ["W1.incl.rs", "B1.incl.rs", "W2.incl.rs", "B2.incl.rs"]:
        src = os.path.join(weights_dir, filename)
        dst = os.path.join(guest_weights_dir, filename)
        shutil.copy2(src, dst)
        print(f"Kopiran {filename} u {guest_weights_dir}")

if __name__ == "__main__":
    convert_weights_to_rust()