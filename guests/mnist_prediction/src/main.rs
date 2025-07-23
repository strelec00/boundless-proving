// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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

fn relu(x: i32) -> i32 {
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
    let mut h1 = [0i32; 64];
    for i in 0..64 {
        let mut sum = weights::B1[i];
        for j in 0..784 {
            sum += weights::W1[j][i] * input[j];
        }
        h1[i] = relu(sum);
    }

    // Layer 2: 64 -> 10 (output layer)
    let mut out = [0i32; 10];
    for i in 0..10 {
        let mut sum = weights::B2[i];
        for j in 0..64 {
            sum += weights::W2[j][i] * h1[j];
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

    // Commit the prediction as U256 for Solidity compatibility
    let prediction = U256::from(predicted_digit);
    env::commit_slice(prediction.abi_encode().as_slice());
}