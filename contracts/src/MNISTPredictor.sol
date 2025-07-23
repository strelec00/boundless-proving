// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pragma solidity ^0.8.20;

import {IRiscZeroVerifier} from "risc0/IRiscZeroVerifier.sol";
import {ImageID} from "./ImageID.sol"; // auto-generated contract after running `cargo build`.
import {IMNISTPredictor} from "./IMNISTPredictor.sol";

/// @title MNIST Neural Network Prediction using RISC Zero.
/// @notice This contract verifies neural network predictions on MNIST images using zero-knowledge proofs.
/// @dev This contract demonstrates offloading neural network inference to a RISC Zero guest
///      while maintaining privacy and verifiability of the computation.
contract MNISTPredictor is IMNISTPredictor {
    /// @notice RISC Zero verifier contract address.
    IRiscZeroVerifier public immutable verifier;

    /// @notice Image ID of the zkVM binary for MNIST prediction.
    ///         This ensures only proofs from the specific MNIST neural network
    ///         guest program are considered valid.
    bytes32 public constant imageId = ImageID.MNIST_PREDICTION_ID;

    /// @notice The last predicted digit (0-2) from MNIST inference.
    uint256 public lastPrediction;

    /// @notice Address of the last user who submitted a prediction.
    address public lastPredictor;

    /// @notice Block number when the last prediction was made.
    uint256 public lastPredictionBlock;

    // Event is inherited from IMNISTPredictor - no need to redeclare

    /// @notice Initialize the contract, binding it to a specified RISC Zero verifier.
    constructor(IRiscZeroVerifier _verifier) {
        verifier = _verifier;
        lastPrediction = 0;
        lastPredictor = address(0);
        lastPredictionBlock = 0;
    }

    /// @notice Submit an MNIST image for prediction. Requires a RISC Zero proof of the neural network computation.
    /// @param imageData Array of 784 pixel values representing a 28x28 MNIST image
    /// @param prediction The predicted digit (0-9) from the neural network
    /// @param seal The RISC Zero proof that the prediction was computed correctly
    function predict(uint256[784] calldata imageData, uint256 prediction, bytes calldata seal) public {
        // Ensure prediction is a valid digit (0-9)
        require(prediction <= 9, "Invalid prediction: must be 0-9");

        // Prepare the journal data for verification
        // The guest program expects a dynamic array of U256 values
        uint256[] memory inputData = new uint256[](784);
        for (uint256 i = 0; i < 784; i++) {
            inputData[i] = imageData[i];
        }

        // Encode the input and expected output for verification
        bytes memory journal = abi.encode(prediction);

        // Verify the zero-knowledge proof
        verifier.verify(seal, imageId, sha256(journal));

        // Store the prediction results
        lastPrediction = prediction;
        lastPredictor = msg.sender;
        lastPredictionBlock = block.number;

        // Emit event
        emit PredictionMade(msg.sender, prediction, block.number);
    }

    /// @notice Get the last prediction made.
    /// @return The last predicted digit (0-9)
    function getLastPrediction() public view returns (uint256) {
        return lastPrediction;
    }

    /// @notice Get the address of the last predictor.
    /// @return The address of the last user who made a prediction
    function getLastPredictor() public view returns (address) {
        return lastPredictor;
    }

    /// @notice Get the block number of the last prediction.
    /// @return The block number when the last prediction was made
    function getLastPredictionBlock() public view returns (uint256) {
        return lastPredictionBlock;
    }
}