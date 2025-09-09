# ZK-AI Inference on Ethereum via Boundless & RISC Zero

This project demonstrates **verifiable AI inference on-chain** using [RISC Zero](https://www.risczero.com/) and [Boundless](https://docs.beboundless.xyz/). A small neural network trained on MNIST is executed in a zero-knowledge VM (zkVM), and the resulting prediction is verified on the Ethereum blockchain via a smart contract.

The model is run off-chain by Prover inside a zkVM, and a **zero-knowledge proof (seal)** is submitted to a Sepolia Ethereum smart contract to **trustlessly verify** the prediction.

## What This Project Does

- Runs a trained neural network (MNIST digit recognizer) in a zkVM
- Produces a ZK proof of the model's prediction
- Submits the prediction and proof to an Ethereum smart contract (`MNISTPredictor`) for on-chain verification
- Stores the verified prediction on-chain

## Quick Start

### CLI-Only Approach (No UI)

#### 1. Install Dependencies

Install RISC Zero and Boundless CLI tooling:

```bash
curl -L https://risczero.com/install | bash
rzup install
```

#### 2. Clone the Repository

```bash
git clone https://github.com/strelec00/boundless-proving
cd boundless-proving
```

#### 3. Set Environment Variables

Make sure your environment is configured with a valid Ethereum private key and RPC URL (e.g., for Sepolia testnet):

```bash
export RPC_URL="https://ethereum-sepolia-rpc.publicnode.com"
export PRIVATE_KEY="your_private_key_here"
export MNIST_PREDICTOR_ADDRESS="0xc64BF4167651Dd458BAd19Eac5e25700F05002aC"
```

You need to have Pinata JWT because it uploads the zkVM guest binary to IPFS for the Boundless network to access and execute:

```bash
export PINATA_JWT="your_pinata_jwt_token"
```

#### 4. Run the Inference App

You can use either a sample MNIST image or provide your own using the frontend, or use templates from given samples and change the 28x28 matrix to your liking.

**Option A: Run with Sample Image**
```bash
RUST_LOG=info cargo run --bin app -- --image-file ./image2.rs
```

**Option B: Run with Custom Image File**
```bash
RUST_LOG=info cargo run --bin app -- --image-file ./your_image.rs
```

### UI Approach (With Frontend)

Set environmental variables from the CLI section above, then:

**Start Frontend:**
```bash
cd boundless-proving/
cd front-end
npm install
npm start
```

**Start Backend (in another terminal):**
```bash
cd boundless-proving/
cd mnist-predictor-backend/
npm run dev
```

**Use the Web Interface:**
1. Open your browser to the frontend URL (typically `http://localhost:3000`)
2. Draw a number on the canvas provided
3. Click "Predict" button to submit for ZK inference
4. The prediction will be processed and verified on-chain

You can verify the proof was submitted by checking the smart contract on Sepolia: https://sepolia.etherscan.io/address/0xc64BF4167651Dd458BAd19Eac5e25700F05002aC

The UI provides a complete web interface for drawing and predicting digits, while the CLI approach gives you direct command-line control over the inference process.

## How It Works

1. The guest binary performs inference using hardcoded weights
2. The prediction result is committed to the journal in the zkVM
3. A seal (ZK proof) and journal (output) are submitted via Boundless to Ethereum
4. The `MNISTPredictor` smart contract verifies the seal and stores the result

## Smart Contracts

- `MNISTPredictor.sol`: Accepts predictions, verifies ZK proofs, and stores results
- Auto-generated interfaces via `alloy` are used in the Rust client

## Resources

- [RISC Zero](https://www.risczero.com/)
- [Boundless Market](https://docs.beboundless.xyz/)

## License

Apache-2.0 © 2024 RISC Zero & Contributors
