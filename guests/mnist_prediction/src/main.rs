use std::io::Read;
use alloy_primitives::U256;
use alloy_sol_types::SolValue;
use risc0_zkvm::guest::env;

mod weights {
    include!("weights/CONV1_WEIGHTS.incl.rs");
    include!("weights/CONV1_BIAS.incl.rs");
    include!("weights/CONV2_WEIGHTS.incl.rs");
    include!("weights/CONV2_BIAS.incl.rs");
    include!("weights/FC1_WEIGHTS.incl.rs");
    include!("weights/FC1_BIAS.incl.rs");
    include!("weights/FC2_WEIGHTS.incl.rs");
    include!("weights/FC2_BIAS.incl.rs");
    include!("weights/MODEL_CONSTANTS.incl.rs");
}

const SCALE: i32 = weights::SCALE;

fn relu(x: i64) -> i64 {
    x.max(0)
}

// Simple 2D convolution with ReLU activation
fn conv2d_layer(
    input: &[i64],       // Flattened input: [channels * height * width]
    weights: &[i32],     // Flattened weights
    bias: &[i32],        // Bias
    in_channels: usize,
    out_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_size: usize,
    padding: usize,
) -> Vec<i64> {
    let output_h = input_h; // Same size due to padding
    let output_w = input_w;
    let mut output = vec![0i64; out_channels * output_h * output_w];

    for out_ch in 0..out_channels {
        for y in 0..output_h {
            for x in 0..output_w {
                let mut sum = bias[out_ch] as i64;

                for in_ch in 0..in_channels {
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let in_y = y as i32 + ky as i32 - padding as i32;
                            let in_x = x as i32 + kx as i32 - padding as i32;

                            if in_y >= 0 && in_y < input_h as i32 && in_x >= 0 && in_x < input_w as i32 {
                                // Input index: in_ch * input_h * input_w + in_y * input_w + in_x
                                let input_idx = in_ch * input_h * input_w +
                                    (in_y as usize) * input_w + (in_x as usize);
                                let input_val = input[input_idx];

                                // Weight index: out_ch * in_channels * kernel_size^2 +
                                //               in_ch * kernel_size^2 + ky * kernel_size + kx
                                let weight_idx = out_ch * in_channels * kernel_size * kernel_size +
                                    in_ch * kernel_size * kernel_size +
                                    ky * kernel_size + kx;

                                let weight_val = weights[weight_idx] as i64;
                                sum += input_val * weight_val;
                            }
                        }
                    }
                }

                // Output index: out_ch * output_h * output_w + y * output_w + x
                let output_idx = out_ch * output_h * output_w + y * output_w + x;
                output[output_idx] = relu(sum);
            }
        }
    }

    output
}

// Max pooling 2x2
fn max_pool2d(
    input: &[i64],
    channels: usize,
    input_h: usize,
    input_w: usize,
) -> Vec<i64> {
    let pool_size = 2;
    let output_h = input_h / pool_size;
    let output_w = input_w / pool_size;
    let mut output = vec![0i64; channels * output_h * output_w];

    for ch in 0..channels {
        for y in 0..output_h {
            for x in 0..output_w {
                let mut max_val = i64::MIN;

                for py in 0..pool_size {
                    for px in 0..pool_size {
                        let in_y = y * pool_size + py;
                        let in_x = x * pool_size + px;

                        if in_y < input_h && in_x < input_w {
                            let input_idx = ch * input_h * input_w + in_y * input_w + in_x;
                            max_val = max_val.max(input[input_idx]);
                        }
                    }
                }

                let output_idx = ch * output_h * output_w + y * output_w + x;
                output[output_idx] = max_val;
            }
        }
    }

    output
}

// Fully connected layer with ReLU (manual implementation to avoid array bounds issues)
fn fc_layer_relu(
    input: &[i64],
    output_size: usize,
) -> Vec<i64> {
    let mut output = vec![0i64; output_size];

    // Ensure we don't go out of bounds
    assert_eq!(input.len(), 1568, "FC1 input should be 1568 features");
    assert_eq!(output_size, 128, "FC1 output should be 128 features");

    for i in 0..output_size {
        let mut sum = weights::FC1_BIAS[i] as i64;

        // FC1_WEIGHTS is [128][1568], so access as weights[i][j]
        for j in 0..input.len() {
            sum += (weights::FC1_WEIGHTS[i][j] as i64) * input[j] / (SCALE as i64);
        }

        output[i] = relu(sum);
    }

    output
}

// Final fully connected layer (no ReLU)
fn fc_layer_final(
    input: &[i64],
) -> Vec<i64> {
    let mut output = vec![0i64; 10];

    // Ensure we don't go out of bounds
    assert_eq!(input.len(), 128, "FC2 input should be 128 features");

    for i in 0..10 {
        let mut sum = weights::FC2_BIAS[i] as i64;

        // FC2_WEIGHTS is [10][128], so access as weights[i][j]
        for j in 0..input.len() {
            sum += (weights::FC2_WEIGHTS[i][j] as i64) * input[j] / (SCALE as i64);
        }

        output[i] = sum;
    }

    output
}

fn main() {
    // Read and decode input
    let mut input_bytes = Vec::<u8>::new();
    env::stdin().read_to_end(&mut input_bytes).unwrap();
    let input_data = <Vec<U256>>::abi_decode(&input_bytes).unwrap();

    // Convert to flattened format: 1 channel, 28x28
    let mut input = vec![0i64; 784];
    for (i, &val) in input_data.iter().enumerate() {
        if i < 784 {
            // Scale input to match training preprocessing
            input[i] = (val.to::<i32>() as i64) * (SCALE as i64);
        }
    }

    eprintln!("Starting CNN inference on 28x28 image...");
    eprintln!("Weight array dimensions:");
    eprintln!("  CONV1_WEIGHTS len: {}", weights::CONV1_WEIGHTS.len());
    eprintln!("  CONV1_BIAS len: {}", weights::CONV1_BIAS.len());
    eprintln!("  CONV2_WEIGHTS len: {}", weights::CONV2_WEIGHTS.len());
    eprintln!("  CONV2_BIAS len: {}", weights::CONV2_BIAS.len());
    eprintln!("  FC1_WEIGHTS dimensions: {}x{}", weights::FC1_WEIGHTS.len(), weights::FC1_WEIGHTS[0].len());
    eprintln!("  FC1_BIAS len: {}", weights::FC1_BIAS.len());
    eprintln!("  FC2_WEIGHTS dimensions: {}x{}", weights::FC2_WEIGHTS.len(), weights::FC2_WEIGHTS[0].len());
    eprintln!("  FC2_BIAS len: {}", weights::FC2_BIAS.len());

    // Conv1: 1x28x28 -> 16x28x28 (kernel=5, padding=2)
    let conv1_out = conv2d_layer(
        &input,
        &weights::CONV1_WEIGHTS,
        &weights::CONV1_BIAS,
        1,  // input channels
        16, // output channels
        28, // input height
        28, // input width
        5,  // kernel size
        2,  // padding
    );
    eprintln!("Conv1 completed: 16x28x28 = {} values", conv1_out.len());

    // MaxPool1: 16x28x28 -> 16x14x14
    let pool1_out = max_pool2d(&conv1_out, 16, 28, 28);
    eprintln!("Pool1 completed: 16x14x14 = {} values", pool1_out.len());

    // Conv2: 16x14x14 -> 32x14x14 (kernel=5, padding=2)
    let conv2_out = conv2d_layer(
        &pool1_out,
        &weights::CONV2_WEIGHTS,
        &weights::CONV2_BIAS,
        16, // input channels
        32, // output channels
        14, // input height
        14, // input width
        5,  // kernel size
        2,  // padding
    );
    eprintln!("Conv2 completed: 32x14x14 = {} values", conv2_out.len());

    // MaxPool2: 32x14x14 -> 32x7x7
    let pool2_out = max_pool2d(&conv2_out, 32, 14, 14);
    eprintln!("Pool2 completed: 32x7x7 = {} values", pool2_out.len());

    // FC1: 1568 -> 128 with ReLU
    let fc1_out = fc_layer_relu(&pool2_out, 128);
    eprintln!("FC1 completed: {} values", fc1_out.len());

    // FC2: 128 -> 10 (final output)
    let output = fc_layer_final(&fc1_out);
    eprintln!("FC2 completed: {} values", output.len());

    // Find predicted digit (argmax)
    let (mut predicted_digit, mut max_val) = (0, output[0]);
    for i in 1..10 {
        if output[i] > max_val {
            max_val = output[i];
            predicted_digit = i;
        }
    }

    // Debug output
    eprintln!("CNN Output scores:");
    for i in 0..10 {
        eprintln!("  Digit {}: {}", i, output[i]);
    }
    eprintln!("Predicted digit: {}", predicted_digit);

    // Commit the prediction
    let prediction = U256::from(predicted_digit);
    env::commit_slice(prediction.abi_encode().as_slice());
}