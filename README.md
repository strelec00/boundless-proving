# ZK-AI Inference on Ethereum via Boundless & RISC Zero

This project demonstrates **verifiable AI inference on-chain** using [RISC Zero](https://www.risczero.com/) and [Boundless](https://docs.beboundless.xyz/). A small neural network trained on MNIST is executed in a zero-knowledge VM (zkVM), and the resulting prediction is verified on the Ethereum blockchain via a smart contract.

The model is run off-chain inside a zkVM, and a **zero-knowledge proof (seal)** is submitted to an Ethereum smart contract to **trustlessly verify** the prediction.

## 🧠 What This Project Does

- Runs a trained neural network (MNIST digit recognizer) in a zkVM
- Produces a ZK proof of the model's prediction
- Submits the prediction and proof to an Ethereum smart contract (`MNISTPredictor`) for on-chain verification
- Stores the verified prediction on-chain

## 🎨 Frontend Interface

The project includes a **frontend folder** with an interactive web interface. When you run `npm start` in /front-end folder, it launches a website where users can:

- **Draw digits on a canvas** using mouse or touch input
- **Generate a 28x28 matrix** automatically from their drawing
- The matrix contains **1s where the user drew** and **0s for empty pixels**
- This matrix can then be used as input for the ZK inference process

This provides an intuitive way to test the MNIST digit recognition without needing to prepare image files manually.

## 🚀 Quick Start

### 1. Install Dependencies

Install RISC Zero and Boundless CLI tooling:
```bash
curl -L https://risczero.com/install | bash
rzup install
```

### 2. Clone the Repository

```bash
git clone https://github.com/strelec00/boundless-proving
cd boundless-proving
```

### 3. Set Environment Variables

Make sure your environment is configured with a valid Ethereum private key and RPC URL (e.g., for Sepolia testnet):

```bash
export RPC_URL="https://ethereum-sepolia-rpc.publicnode.com"
export PRIVATE_KEY="your_private_key_here"
export MNIST_PREDICTOR_ADDRESS="deployed_contract_address" or use predeployed address MNIST_PREDICTOR_ADDRESS="0x973E2b3d5996a4C915016887eDd48b7A80C8BdE0"
```

If uploading to IPFS via Pinata:

```bash
export PINATA_JWT="your_pinata_jwt_token"
```

### 4. Run the Frontend

You can use either a sample MNIST image or provide your own.

```bash
cd frontend
npm install
npm start
```
This will open a web browser where you can draw digits and generate the 28x28 matrix.

### 5. Run the Inference App

You can use either a sample MNIST image or provide your own.

#### Option A: Run with Sample Image

```bash
RUST_LOG=info MNIST_PREDICTOR_ADDRESS="0x973E2b3d5996a4C915016887eDd48b7A80C8BdE0" cargo run --bin app -- --sample
```

#### Option B: Run with Custom Image File

```bash
RUST_LOG=info MNIST_PREDICTOR_ADDRESS="0x973E2b3d5996a4C915016887eDd48b7A80C8BdE0" cargo run --bin app -- --image-file ./your_matrix.rs
```

Image file format must be a list of 784 integers (28x28 pixels), either as a JSON array, CSV, or whitespace-separated.

## 🧪 How It Works

1. The guest binary performs inference using hardcoded weights
2. The prediction result is committed to the journal in the zkVM
3. A seal (ZK proof) and journal (output) are submitted via Boundless to Ethereum
4. The `MNISTPredictor` smart contract verifies the seal and stores the result

## 🧰 Development

### Build Contracts & Guests

```bash
forge build
cargo build
```

### Test Contracts and Guest Code

```bash
forge test -vvv
cargo test
```

## 🛠 Deploying the Smart Contract

Deploy the `MNISTPredictor` contract to Sepolia:

```bash
VERIFIER_ADDRESS="your_verifier_address_here"
forge script contracts/scripts/DeployMNISTPredictor.s.sol --rpc-url ${RPC_URL:?} --broadcast -vv
export MNIST_PREDICTOR_ADDRESS="returned_address_from_deploy"
```

## ☁️ Uploading Guest Programs

You can upload the guest binary to IPFS or S3. To use Pinata:

```bash
export PINATA_JWT="your_pinata_jwt"
```

Run the app without the `--program-url` flag, and it will automatically upload and use the resulting URL:

```bash
cargo run --bin app -- --sample
```

Or, if you already uploaded your guest:

```bash
cargo run --bin app -- --sample --program-url https://your.ipfs.link/to/guest
```

## 📄 Smart Contracts

- `MNISTPredictor.sol`: Accepts predictions, verifies ZK proofs, and stores results
- Auto-generated interfaces via `alloy` are used in the Rust client

## ✅ Status

- ✅ ZK Inference
- ✅ Seal Generation
- ✅ Ethereum Verification
- ✅ On-Chain Storage
- ✅ Frontend Interface

## 🧠 Credits

- [RISC Zero](https://www.risczero.com/)
- [Boundless Market](https://docs.beboundless.xyz/)
- Inspired by `boundless-foundry-template`

## 📜 License

Apache-2.0 © 2024 RISC Zero & Contributors
