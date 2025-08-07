// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.20;

import {Script} from "forge-std/Script.sol";
import {console2} from "forge-std/console2.sol";
import {MNISTPredictor} from "../src/MNISTPredictor.sol";
import {IRiscZeroVerifier} from "risc0/IRiscZeroVerifier.sol";

contract DeployScript is Script {
    function run() external {
        // Get the verifier address from environment
        address verifierAddress = vm.envOr("RISC_ZERO_VERIFIER_ADDRESS", address(0));

        if (verifierAddress == address(0)) {
            console2.log("ERROR: RISC_ZERO_VERIFIER_ADDRESS not set");
            console2.log("Please set the environment variable:");
            console2.log("export RISC_ZERO_VERIFIER_ADDRESS=0x...");
            return;
        }

        console2.log("Deploying MNISTPredictor...");
        console2.log("Using verifier at:", verifierAddress);
        console2.log("Deployer:", msg.sender);

        vm.startBroadcast();

        // Deploy the MNISTPredictor contract
        MNISTPredictor predictor = new MNISTPredictor(
            IRiscZeroVerifier(verifierAddress)
        );

        vm.stopBroadcast();

        console2.log("MNISTPredictor deployed at:", address(predictor));
        console2.log(" Save this address for your app:");
        console2.log("   --mnist-predictor-address", address(predictor));

        // Verify the deployment
        console2.log(" Verifying deployment...");
        console2.log("   Image ID:");
        console2.logBytes32(predictor.imageId());
        console2.log("   Verifier:", address(predictor.verifier()));
        console2.log("   Last prediction:", predictor.getLastPrediction());
    }
}