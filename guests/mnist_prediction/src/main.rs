// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::io::Read;
use alloy_primitives::U256;
use alloy_sol_types::SolValue;
use risc0_zkvm::guest::env;

mod weights {
    include!("weights/W1.incl.rs");
    include!("weights/B1.incl.rs");
    include!("weights/W2.incl.rs");
    include!("weights/B2.incl.rs");
}

// Updated scale to match training
const SCALE: i32 = 16384;  // Must match the SCALE in Python

fn relu(x: i64) -> i64 {
    x.max(0)
}

fn main() {
    // Read the input data for this application.
    let mut input_bytes = Vec::<u8>::new();
    env::stdin().read_to_end(&mut input_bytes).unwrap();

    // Decode the input - expecting array of 784 U256 values
    let input_data = <Vec<U256>>::abi_decode(&input_bytes).unwrap();

    // Convert U256 to i32 array for neural network processing
    let mut input: [i32; 784] = [0; 784];
    for (i, &val) in input_data.iter().enumerate() {
        if i < 784 {
            input[i] = val.to::<i32>();
        }
    }

    // Layer 1: 784 -> 64 with ReLU activation
    // Using i64 for intermediate calculations to prevent overflow
    let mut h1 = [0i64; 64];
    for i in 0..64 {
        let mut sum = weights::B1[i] as i64; // Don't multiply by SCALE again
        for j in 0..784 {
            sum += (weights::W1[j][i] as i64) * (input[j] as i64);
        }
        h1[i] = relu(sum); // Already scaled by SCALE once
    }


    // Layer 2: 64 -> 10 (output layer)
    let mut out = [0i64; 10];
    for i in 0..10 {
        let mut sum = weights::B2[i] as i64;
        for j in 0..64 {
            sum += (weights::W2[j][i] as i64) * h1[j] / (SCALE as i64);
        }
        out[i] = sum;
    }


    // Find predicted digit (argmax)
    let (mut predicted_digit, mut max_val) = (0, out[0]);
    for i in 1..10 {
        if out[i] > max_val {
            max_val = out[i];
            predicted_digit = i;
        }
    }

    // Debug output (optional - remove in production)
    eprintln!("Output scores (scaled):");
    for i in 0..10 {
        eprintln!("  Digit {}: {}", i, out[i]);
    }
    eprintln!("Predicted digit: {}", predicted_digit);

    // Commit the prediction as U256 for Solidity compatibility
    let prediction = U256::from(predicted_digit);
    env::commit_slice(prediction.abi_encode().as_slice());
}