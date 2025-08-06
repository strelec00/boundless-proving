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

use std::fs;
use std::time::Duration;

use crate::mnist_predictor::IMNISTPredictor::IMNISTPredictorInstance;
use alloy::{primitives::{Address, U256}, signers::local::PrivateKeySigner, sol_types::SolValue};
use anyhow::{bail, Context, Result};
use boundless_market::{Client, Deployment, StorageProviderConfig};
use clap::Parser;
use guests::MNIST_PREDICTION_ELF;
use url::Url;

/// Timeout for the transaction to be confirmed.
pub const TX_TIMEOUT: Duration = Duration::from_secs(30);

mod mnist_predictor {
    alloy::sol!(
        #![sol(rpc, all_derives)]
        "../contracts/src/IMNISTPredictor.sol"
    );
}

mod sample {
    // Include a sample MNIST image for testing
    include!("input/sample_input.rs");
}
/// Arguments for the MNIST prediction application
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// MNIST image input: either "--sample" or path to file
    #[clap(long)]
    sample: bool,
    /// Path to MNIST image file (784 integers)
    #[clap(long)]
    image_file: Option<String>,
    /// URL of the Ethereum RPC endpoint.
    #[clap(short, long, env)]
    rpc_url: Url,
    /// Private key used to interact with the MNISTPredictor contract and the Boundless Market.
    #[clap(long, env)]
    private_key: PrivateKeySigner,
    /// Address of the MNISTPredictor contract.
    #[clap(short, long, env)]
    mnist_predictor_address: Address,
    /// URL where provers can download the program to be proven.
    #[clap(long, env)]
    program_url: Option<Url>,
    /// Submit the request offchain via the provided order stream service url.
    #[clap(short, long, requires = "order_stream_url")]
    offchain: bool,
    /// Configuration for the StorageProvider to use for uploading programs and inputs.
    #[clap(flatten, next_help_heading = "Storage Provider")]
    storage_config: StorageProviderConfig,
    /// Deployment of the Boundless contracts and services to use.
    ///
    /// Will be automatically resolved from the connected chain ID if unspecified.
    #[clap(flatten, next_help_heading = "Boundless Market Deployment")]
    deployment: Option<Deployment>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match dotenvy::dotenv() {
        Ok(path) => tracing::debug!("Loaded environment variables from {:?}", path),
        Err(e) if e.not_found() => tracing::debug!("No .env file found"),
        Err(e) => bail!("failed to load .env file: {}", e),
    }
    let args = Args::parse();

    // Get the input data
    let input_data = if args.sample {
        tracing::info!("Using sample MNIST image...");
        sample::SAMPLE.to_vec()
    } else if let Some(path) = args.image_file {
        tracing::info!("Loading MNIST image from: {}", path);
        load_mnist_from_file(&path)?
    } else {
        bail!("Must specify either --sample or --image-file");
    };

    // Ensure we have exactly 784 values for MNIST (28x28 image)
    if input_data.len() != 784 {
        bail!("MNIST image must contain exactly 784 pixel values (28x28)");
    }

    // Convert to U256 for Solidity compatibility
    let input_u256: Vec<U256> = input_data.iter()
        .map(|&val| U256::from(val.max(0) as u32))
        .collect();

    // Create a Boundless client from the provided parameters.
    let client = Client::builder()
        .with_rpc_url(args.rpc_url)
        .with_deployment(args.deployment)
        .with_storage_provider_config(&args.storage_config)?
        .with_private_key(args.private_key)
        .build()
        .await
        .context("failed to build boundless client")?;

    // Encode the input for the guest program
    tracing::info!("MNIST image loaded with {} pixels", input_data.len());
    let input_bytes = input_u256.abi_encode();

    // Build the request based on whether program URL is provided
    let request = if let Some(program_url) = args.program_url {
        // Use the provided URL
        client
            .new_request()
            .with_program_url(program_url)?
            .with_stdin(input_bytes.clone())
    } else {
        client
            .new_request()
            .with_program(MNIST_PREDICTION_ELF)
            .with_stdin(input_bytes)
    };

    let (request_id, expires_at) = client.submit_onchain(request).await?;

    // Wait for the request to be fulfilled. The market will return the journal and seal.
    tracing::info!("Waiting for request {:x} to be fulfilled", request_id);
    let (journal, seal) = client
        .wait_for_request_fulfillment(
            request_id,
            Duration::from_secs(5), // check every 5 seconds
            expires_at,
        )
        .await?;
    tracing::info!("Request {:x} fulfilled", request_id);

    // Extract the prediction from the journal
    let prediction: U256 = U256::abi_decode(&journal)?;
    tracing::info!("Predicted digit: {}", prediction);

    // We interact with the MNISTPredictor contract by calling the predict function with our
    // image data, prediction, and the seal (i.e. proof) returned by the market.
    let mnist_predictor = IMNISTPredictorInstance::new(args.mnist_predictor_address, client.provider().clone());

    // Convert input data to fixed-size array for contract call
    let image_array: [U256; 784] = input_u256.try_into().map_err(|_| {
        anyhow::anyhow!("Failed to convert input to fixed-size array")
    })?;

    let call_predict = mnist_predictor
        .predict(image_array, prediction, seal)
        .from(client.caller());

    // By calling the predict function, we verify the seal against the published roots
    // of the MNISTVerifier contract.
    tracing::info!("Calling MNISTPredictor predict function");
    let pending_tx = call_predict.send().await.context("failed to broadcast tx")?;
    tracing::info!("Broadcasting tx {}", pending_tx.tx_hash());
    let tx_hash = pending_tx
        .with_timeout(Some(TX_TIMEOUT))
        .watch()
        .await
        .context("failed to confirm tx")?;
    tracing::info!("Tx {:?} confirmed", tx_hash);

    // Query the value stored at the MNISTPredictor address to check it was set correctly
    let stored_prediction = mnist_predictor
        .getLastPrediction()
        .call()
        .await
        .context("failed to get prediction from contract")?;
    tracing::info!(
        "The prediction for contract at address: {:?} is set to {:?}",
        args.mnist_predictor_address,
        stored_prediction
    );

    Ok(())
}

fn load_mnist_from_file(path: &str) -> Result<Vec<i32>> {
    let contents = fs::read_to_string(path)?;

    // Parse the file content - support different formats
    let input_data: Vec<i32> = if contents.contains("pub const") && contents.contains("[i32; 784]") {
        // Rust array format - extract the array part
        tracing::info!("Parsing Rust array format file");

        let start = contents.find("= [")
            .context("No array start '= [' found in Rust file")?
            + 3; // Skip "= ["

        let end = contents.rfind("];")
            .context("No array end '];' found in Rust file")?;

        let array_content = &contents[start..end];

        // Parse comma-separated values, handling whitespace and empty entries
        array_content
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.parse::<i32>()
                        .with_context(|| format!("Failed to parse number: '{}'", trimmed)))
                }
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if contents.trim().starts_with('[') {
        // JSON array format
        serde_json::from_str(&contents)?
    } else if contents.contains(',') {
        // Comma-separated values
        contents
            .split(',')
            .map(|s| s.trim().parse::<i32>())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Space-separated values
        contents
            .split_whitespace()
            .map(|s| s.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(input_data)
}