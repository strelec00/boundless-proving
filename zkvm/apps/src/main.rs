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

use std::{fs, io};
use std::path::Path;
use std::time::{Duration, Instant};

use alloy::{primitives::{Address, U256, Bytes}, signers::local::PrivateKeySigner, sol_types::SolValue};
use alloy::consensus::private::serde::{Deserialize, Serialize};
use alloy::core::sol;
use alloy::providers::Provider;
use anyhow::{bail, Context, Result};
use boundless_market::{Client, Deployment, StorageProviderConfig};
use clap::Parser;
use guests::MNIST_PREDICTION_ELF;
use url::Url;

/// Timeout for the transaction to be confirmed.
pub const TX_TIMEOUT: Duration = Duration::from_secs(30);

use tokio::time::timeout;
use tracing::{debug, info, warn};

mod mnist_predictor {
    alloy::sol!(
        #![sol(rpc, all_derives)]
        "../contracts/src/IMNISTPredictor.sol"
    );
}

sol! {
    #[allow(missing_docs)]
    #[sol(rpc)]
    contract IMNISTPredictor {
        function predict(uint256[784] calldata imageData, uint256 prediction, bytes calldata seal) external;
        function getLastPrediction() external view returns (uint256);
        function getLastPredictor() external view returns (address);
        function getLastPredictionBlock() external view returns (uint256);

        event PredictionMade(address indexed predictor, uint256 prediction, uint256 blockNumber);
    }
}

/// Arguments for the MNIST prediction application
#[derive(Parser, Debug)]
#[command(name = "mnist-zk-predictor")]
#[command(about = "Zero-Knowledge MNIST Digit Prediction System using Boundless Market")]
struct Args {
    /// Use sample MNIST image for testing
    #[arg(long)]
    sample: bool,

    /// Path to JSON file containing 784 pixel values (0-255)
    #[arg(long, conflicts_with = "sample")]
    image_file: Option<String>,

    /// Raw image data as comma-separated values (0-255)
    #[arg(long, conflicts_with_all = ["sample", "image_file"])]
    image_data: Option<String>,

    /// Address of deployed MNISTPredictor contract
    #[arg(long)]
    mnist_predictor_address: String,

    /// RPC URL for Ethereum network (defaults to Sepolia)
    #[arg(long, default_value = "https://ethereum-sepolia-rpc.publicnode.com/")]
    rpc_url: String,

    /// Private key for transaction signing (or set PRIVATE_KEY env var)
    #[arg(long, env = "PRIVATE_KEY")]
    private_key: Option<String>,

    /// URL where provers can download the program to be proven
    #[arg(long, env)]
    program_url: Option<Url>,

    /// Submit the request offchain via the provided order stream service url
    #[arg(short, long)]
    offchain: bool,

    /// Configuration for the StorageProvider to use for uploading programs and inputs
    #[clap(flatten, next_help_heading = "Storage Provider")]
    storage_config: StorageProviderConfig,

    /// Deployment of the Boundless contracts and services to use
    #[clap(flatten, next_help_heading = "Boundless Market Deployment")]
    deployment: Option<Deployment>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Timeout for ZK proof generation (seconds) via Boundless
    #[arg(long, default_value = "600")]
    proof_timeout: u64,

    /// Skip blockchain submission (useful for testing)
    #[arg(long)]
    dry_run: bool,

    /// Output results to JSON file
    #[arg(long)]
    output_file: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionResult {
    prediction: u32,
    confidence: f32,
    proof_generation_time_ms: u128,
    request_id: Option<String>,
    transaction_hash: Option<String>,
    gas_used: Option<u64>,
    block_number: Option<u64>,
    image_stats: ImageStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageStats {
    min_pixel: u32,
    max_pixel: u32,
    mean_pixel: f32,
    non_zero_pixels: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageInput {
    pixels: Vec<u32>,
    metadata: Option<serde_json::Value>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("{}={}", env!("CARGO_PKG_NAME"), log_level))
        .init();

    // Load .env file if available
    match dotenvy::dotenv() {
        Ok(path) => tracing::debug!("Loaded environment variables from {:?}", path),
        Err(e) if e.not_found() => tracing::debug!("No .env file found"),
        Err(e) => bail!("failed to load .env file: {}", e),
    }

    info!("🚀 Starting ZK-MNIST Prediction System with Boundless Market");
    info!("📋 Contract Address: {}", args.mnist_predictor_address);
    info!("🌐 RPC URL: {}", args.rpc_url);

    // Load and validate image data
    let image_data = match load_image_data(&args).await {
        Ok(data) => data,
        Err(e) => {
            eprintln!("❌ Failed to load image data: {}", e);
            return Err(e);
        }
    };

    let image_stats = calculate_image_stats(&image_data);
    info!("📊 Image Stats: {:?}", image_stats);

    if let Err(e) = validate_image_data(&image_data) {
        eprintln!("❌ Image validation failed: {}", e);
        return Err(e);
    }

    // Parse private key
    let private_key: PrivateKeySigner = args.private_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Private key required"))?
        .parse()?;

    // Parse RPC URL
    let rpc_url: Url = args.rpc_url.parse()?;

    // Create Boundless client
    info!("🔗 Connecting to Boundless Market...");
    let client = Client::builder()
        .with_rpc_url(rpc_url.clone())
        .with_deployment(args.deployment)
        .with_storage_provider_config(&args.storage_config)?
        .with_private_key(private_key.clone())
        .build()
        .await
        .context("failed to build boundless client")?;

    // Generate ZK proof via Boundless Market
    info!("🔐 Submitting proof request to Boundless Market...");
    let proof_start = Instant::now();

    let (prediction, seal, request_id) = match timeout(
        Duration::from_secs(args.proof_timeout),
        generate_proof_via_boundless(&client, image_data.clone(), args.program_url)
    ).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            eprintln!("❌ Boundless proof generation failed: {}", e);
            return Err(e);
        }
        Err(_) => {
            eprintln!("❌ Boundless proof generation timed out");
            return Err(anyhow::anyhow!("Boundless proof generation timed out"));
        }
    };

    let proof_time = proof_start.elapsed();
    info!("✅ ZK proof generated via Boundless in {:.2}s", proof_time.as_secs_f32());
    info!("🎯 Predicted digit: {}", prediction);
    info!("🆔 Request ID: {}", request_id);

    // Calculate confidence
    let confidence = calculate_confidence(&image_data, prediction);
    info!("📈 Confidence: {:.1}%", confidence * 100.0);

    let mut result = PredictionResult {
        prediction,
        confidence,
        proof_generation_time_ms: proof_time.as_millis(),
        request_id: Some(request_id),
        transaction_hash: None,
        gas_used: None,
        block_number: None,
        image_stats,
    };

    // Submit to blockchain (unless dry run)
    if !args.dry_run {
        info!("📤 Submitting prediction to blockchain...");
        match submit_to_blockchain_with_boundless(
            &client,
            &args.mnist_predictor_address,
            image_data,
            prediction,
            seal,
        ).await {
            Ok(tx_result) => {
                result.transaction_hash = Some(tx_result.tx_hash);
                result.gas_used = tx_result.gas_used;
                result.block_number = tx_result.block_number;

                info!("🎉 Transaction successful!");
                info!("📄 TX Hash: {}", result.transaction_hash.clone().unwrap());
                if let Some(gas) = tx_result.gas_used {
                    info!("⛽ Gas Used: {}", gas);
                }
            }
            Err(e) => {
                eprintln!("❌ Blockchain submission failed: {}", e);
                return Err(e);
            }
        }
    } else {
        info!("🏃 Dry run mode - skipping blockchain submission");
    }

    // Output results
    if let Some(output_path) = args.output_file {
        if let Err(e) = save_results_to_file(&result, &output_path) {
            warn!("⚠️  Failed to save results to file: {}", e);
        } else {
            info!("💾 Results saved to: {}", output_path);
        }
    }

    print_summary(&result);
    info!("✨ Prediction completed successfully!");

    Ok(())
}

async fn load_image_data(args: &Args) -> Result<Vec<u32>> {
    if args.sample {
        info!("📷 Loading sample MNIST image...");
        Ok(load_sample_image())
    } else if let Some(file_path) = &args.image_file {
        info!("📁 Loading image from file: {}", file_path);
        load_image_from_file(file_path)
    } else if let Some(data_str) = &args.image_data {
        info!("📝 Parsing image data from command line...");
        parse_image_data_string(data_str)
    } else {
        // Interactive mode
        info!("🖊️  Interactive mode - please input image data");
        load_image_interactive().await
    }
}

fn load_sample_image() -> Vec<u32> {
    // Sample MNIST digit "2" (28x28 = 784 pixels)
    let mut pixels = vec![0u32; 784];

    // Draw a simple "2" pattern
    let pattern = vec![
        (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
        (9, 9), (9, 10), (8, 11), (7, 12), (6, 13),
        (5, 14), (6, 15), (7, 16), (8, 17), (9, 18),
        (5, 19), (6, 19), (7, 19), (8, 19), (9, 19),
    ];

    for (row, col) in pattern {
        if row < 28 && col < 28 {
            pixels[row * 28 + col] = 1; // Use 0 or 1 instead of 0-255
        }
    }

    pixels
}

fn load_image_from_file(file_path: &str) -> Result<Vec<u32>> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Err(anyhow::anyhow!("Image file not found: {}", file_path));
    }

    let content = fs::read_to_string(path)
        .context("Failed to read image file")?;

    // Try to parse as JSON first
    if let Ok(input) = serde_json::from_str::<ImageInput>(&content) {
        // Normalize pixel values to 0-1 range if they're in 0-255 range
        let normalized_pixels = normalize_pixel_values(input.pixels);
        return Ok(normalized_pixels);
    }

    // Try to parse as simple JSON array
    if let Ok(pixels) = serde_json::from_str::<Vec<u32>>(&content) {
        let normalized_pixels = normalize_pixel_values(pixels);
        return Ok(normalized_pixels);
    }

    // Try to parse as comma-separated values
    let pixels = parse_image_data_string(&content)?;
    let normalized_pixels = normalize_pixel_values(pixels);
    Ok(normalized_pixels)
}

fn normalize_pixel_values(pixels: Vec<u32>) -> Vec<u32> {
    // Check if pixels are in 0-255 range and normalize to 0-1
    let max_value = pixels.iter().max().cloned().unwrap_or(0);

    if max_value > 1 {
        // Normalize from 0-255 to 0-1
        pixels.into_iter()
            .map(|p| if p > 128 { 1 } else { 0 })
            .collect()
    } else {
        // Already in 0-1 range
        pixels
    }
}

fn parse_image_data_string(data_str: &str) -> Result<Vec<u32>> {
    let pixels: Result<Vec<u32>> = data_str
        .trim()
        .split(',')
        .enumerate()
        .map(|(i, s)| {
            s.trim().parse::<u32>()
                .with_context(|| format!("Invalid pixel value at position {}: '{}'", i, s.trim()))
        })
        .collect();

    pixels
}

async fn load_image_interactive() -> Result<Vec<u32>> {
    println!("Enter 784 pixel values (0-255) separated by commas:");
    println!("Or enter path to JSON file:");

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    // Check if it's a file path
    if Path::new(input).exists() {
        load_image_from_file(input)
    } else {
        let pixels = parse_image_data_string(input)?;
        let normalized_pixels = normalize_pixel_values(pixels);
        Ok(normalized_pixels)
    }
}

fn calculate_image_stats(pixels: &[u32]) -> ImageStats {
    let min_pixel = *pixels.iter().min().unwrap_or(&0);
    let max_pixel = *pixels.iter().max().unwrap_or(&1);
    let mean_pixel = pixels.iter().sum::<u32>() as f32 / pixels.len() as f32;
    let non_zero_pixels = pixels.iter().filter(|&&p| p > 0).count();

    ImageStats {
        min_pixel,
        max_pixel,
        mean_pixel,
        non_zero_pixels,
    }
}

fn validate_image_data(pixels: &[u32]) -> Result<()> {
    if pixels.len() != 784 {
        return Err(anyhow::anyhow!(
            "Invalid image data: expected 784 pixels (28x28), got {}",
            pixels.len()
        ));
    }

    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel > 1 {
            return Err(anyhow::anyhow!(
                "Invalid pixel value at position {}: {} (must be 0 or 1 after normalization)",
                i, pixel
            ));
        }
    }

    let non_zero_count = pixels.iter().filter(|&&p| p > 0).count();
    if non_zero_count < 10 {
        warn!("⚠️  Image appears to be mostly empty ({} non-zero pixels)", non_zero_count);
    }

    Ok(())
}

// NEW: Generate proof using Boundless Market instead of local proving
async fn generate_proof_via_boundless(
    client: &Client,
    image_data: Vec<u32>,
    program_url: Option<Url>
) -> Result<(u32, Vec<u8>, String)> {
    debug!("🔧 Preparing data for Boundless Market...");

    // Convert to U256 for Solidity compatibility
    let input_u256: Vec<U256> = image_data.iter()
        .map(|&val| U256::from(val))
        .collect();

    debug!("📊 Image data summary: {} pixels converted to U256", input_u256.len());

    // Encode the input for the guest program
    let input_bytes = input_u256.abi_encode();

    // Build the request based on whether program URL is provided
    let request = if let Some(program_url) = program_url {
        debug!("📡 Using provided program URL: {}", program_url);
        client
            .new_request()
            .with_program_url(program_url)?
            .with_stdin(input_bytes.clone())
    } else {
        debug!("📦 Using embedded program ELF");
        client
            .new_request()
            .with_program(MNIST_PREDICTION_ELF)
            .with_stdin(input_bytes)
    };

    // Submit the request to Boundless Market
    info!("📤 Submitting request to Boundless Market...");
    let (request_id, expires_at) = client.submit_onchain(request).await?;

    let request_id_str = format!("{:x}", request_id);
    info!("🆔 Request submitted with ID: {}", request_id_str);
    info!("⏰ Request expires at: {:?}", expires_at);

    // Wait for the request to be fulfilled
    info!("⏳ Waiting for request fulfillment...");
    let (journal, seal) = client
        .wait_for_request_fulfillment(
            request_id,
            Duration::from_secs(5), // check every 5 seconds
            expires_at,
        )
        .await
        .context("Failed to get proof from Boundless Market")?;

    info!("✅ Request {} fulfilled by Boundless Market", request_id_str);

    // Extract the prediction from the journal
    let prediction: U256 = U256::abi_decode(&journal)
        .context("Failed to decode prediction from journal")?;

    let prediction_u32 = prediction.to::<u32>();
    debug!("🎯 Decoded prediction: {}", prediction_u32);

    Ok((prediction_u32, seal.to_vec(), request_id_str))
}

fn calculate_confidence(_image_data: &[u32], _prediction: u32) -> f32 {
    // This is a simplified confidence calculation
    // In a real system, this would come from your ML model

    let non_zero_pixels = _image_data.iter().filter(|&&p| p > 0).count();
    let pixel_variance = {
        let mean = _image_data.iter().sum::<u32>() as f32 / _image_data.len() as f32;
        let variance_sum: f32 = _image_data.iter()
            .map(|&p| (p as f32 - mean).powi(2))
            .sum();
        variance_sum / _image_data.len() as f32
    };

    // Simple heuristic: more non-zero pixels and higher variance = higher confidence
    let base_confidence = (non_zero_pixels as f32 / 784.0).min(1.0);
    let variance_factor = (pixel_variance / 1.0).min(1.0); // Adjusted for 0-1 range

    (base_confidence * 0.7 + variance_factor * 0.3).max(0.1).min(0.99)
}

#[derive(Debug)]
struct TransactionResult {
    tx_hash: String,
    gas_used: Option<u64>,
    block_number: Option<u64>,
}

// NEW: Submit to blockchain using Boundless client
async fn submit_to_blockchain_with_boundless(
    client: &Client,
    contract_address: &str,
    image_data: Vec<u32>,
    prediction: u32,
    seal: Vec<u8>,
) -> Result<TransactionResult> {
    debug!("🔗 Preparing blockchain transaction...");

    let contract_addr: Address = contract_address.parse()?;
    let mnist_predictor = mnist_predictor::IMNISTPredictor::IMNISTPredictorInstance::new(
        contract_addr,
        client.provider().clone()
    );

    debug!("📝 Preparing contract call...");

    // Convert image data to U256 and then to fixed array
    let input_u256: Vec<U256> = image_data.iter()
        .map(|&pixel| U256::from(pixel))
        .collect();

    let image_array: [U256; 784] = input_u256.try_into().map_err(|_| {
        anyhow::anyhow!("Failed to convert input to fixed-size array")
    })?;

    debug!("📤 Submitting transaction...");

    let call_predict = mnist_predictor
        .predict(image_array, U256::from(prediction), Bytes::from(seal))
        .from(client.caller());

    // Submit the transaction
    let pending_tx = call_predict.send().await
        .context("Failed to broadcast transaction")?;

    info!("📡 Broadcasting tx {}", pending_tx.tx_hash());

    // Wait for confirmation and get the transaction hash
    let tx_hash = pending_tx
        .with_timeout(Some(TX_TIMEOUT))
        .watch()
        .await
        .context("Failed to confirm transaction")?;

    info!("✅ Transaction confirmed");

    // Get the full receipt using the transaction hash
    let receipt = client.provider()
        .get_transaction_receipt(tx_hash)
        .await
        .context("Failed to get transaction receipt")?
        .ok_or_else(|| anyhow::anyhow!("Transaction receipt not found"))?;

    Ok(TransactionResult {
        tx_hash: format!("{:?}", tx_hash),
        gas_used: Some(receipt.gas_used),
        block_number: receipt.block_number,
    })
}

fn save_results_to_file(result: &PredictionResult, file_path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(result)?;
    fs::write(file_path, json)?;
    Ok(())
}

fn print_summary(result: &PredictionResult) {
    println!("\n🎯 === PREDICTION SUMMARY ===");
    println!("   Predicted Digit: {}", result.prediction);
    println!("   Confidence: {:.1}%", result.confidence * 100.0);
    println!("   Proof Generation: {:.2}s", result.proof_generation_time_ms as f32 / 1000.0);

    if let Some(ref request_id) = result.request_id {
        println!("   Boundless Request ID: {}", request_id);
    }

    if let Some(ref tx_hash) = result.transaction_hash {
        println!("   Transaction: {}", tx_hash);
    }

    if let Some(gas) = result.gas_used {
        println!("   Gas Used: {}", gas);
    }

    println!("   Non-zero Pixels: {}/784", result.image_stats.non_zero_pixels);
    println!("===============================\n");
}