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

pragma solidity ^0.8.20;

/// @title Interface for MNIST Neural Network Prediction contract
/// @notice Interface for verifying neural network predictions on MNIST images using zero-knowledge proofs
interface IMNISTPredictor {
    /// @notice Event emitted when a new prediction is made.
    event PredictionMade(address indexed predictor, uint256 prediction, uint256 blockNumber);

    /// @notice Submit an MNIST image for prediction. Requires a RISC Zero proof of the neural network computation.
    /// @param imageData Array of 784 pixel values representing a 28x28 MNIST image
    /// @param prediction The predicted digit (0-9) from the neural network
    /// @param seal The RISC Zero proof that the prediction was computed correctly
    function predict(uint256[784] calldata imageData, uint256 prediction, bytes calldata seal) external;

    /// @notice Get the last prediction made.
    /// @return The last predicted digit (0-9)
    function getLastPrediction() external view returns (uint256);

    /// @notice Get the address of the last predictor.
    /// @return The address of the last user who made a prediction
    function getLastPredictor() external view returns (address);

    /// @notice Get the block number of the last prediction.
    /// @return The block number when the last prediction was made
    function getLastPredictionBlock() external view returns (uint256);
}