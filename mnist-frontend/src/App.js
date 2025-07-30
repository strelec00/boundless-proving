import React, { useState, useRef, useEffect } from "react";
import {
  Brain,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2,
  Wallet,
} from "lucide-react";

const MNISTPredictor = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [zkProofStatus, setZkProofStatus] = useState("idle");
  const [transactionHash, setTransactionHash] = useState("");
  const [error, setError] = useState("");
  const [walletConnected, setWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState("");

  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:3001";
  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
  }, []);

  // Connect wallet
  const connectWallet = async () => {
    if (typeof window.ethereum !== "undefined") {
      try {
        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });
        setWalletConnected(true);
        setWalletAddress(accounts[0]); // Postavlja adresu prvog account-a
        setError("");
      } catch (err) {
        setError("Failed to connect wallet");
      }
    } else {
      setError("Please install MetaMask");
    }
  };

  // Funkcija za formatiranje wallet adrese (skraćuje je za prikaz)
  const formatWalletAddress = (address) => {
    if (!address) return "";
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  // Funkcija za kopiranje adrese u clipboard
  const copyAddressToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(walletAddress);
      // Možete dodati toast notifikaciju ovdje
    } catch (err) {
      console.error("Failed to copy address:", err);
    }
  };

  const getEventPosition = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    if (event.touches && event.touches.length > 0) {
      return {
        x: event.touches[0].clientX - rect.left,
        y: event.touches[0].clientY - rect.top,
      };
    }

    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  };

  const startDrawing = (event) => {
    event.preventDefault();
    setIsDrawing(true);
    const pos = getEventPosition(event);
    drawAt(pos.x, pos.y);
  };

  const draw = (event) => {
    if (!isDrawing) return;
    event.preventDefault();

    const pos = getEventPosition(event);
    drawAt(pos.x, pos.y);
  };

  const drawAt = (x, y) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Draw a larger, smoother circle for better digit recognition
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fill();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    setPrediction(null);
    setConfidence(null);
    setZkProofStatus("idle");
    setTransactionHash("");
    setError("");
  };

  // Convert canvas to 28x28 binary matrix (0s and 1s only)
  const getImageData = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Create temp canvas for proper downsampling
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext("2d");

    // Disable smoothing for crisp pixels
    tempCtx.imageSmoothingEnabled = false;
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const binaryPixels = [];

    // Convert to binary with clear threshold
    for (let i = 0; i < imageData.data.length; i += 4) {
      const gray =
        (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
      binaryPixels.push(gray > 128 ? 1 : 0);
    }

    return binaryPixels;
  };

  // Updated submitPrediction function with better error handling and timeout
  const submitPrediction = async () => {
    if (!walletConnected) {
      setError("Please connect your wallet first");
      return;
    }

    setIsProcessing(true);
    setError("");
    setZkProofStatus("generating");

    try {
      const imageData = getImageData();

      // Validate data
      const nonZeroCount = imageData.filter((p) => p === 1).length;
      if (nonZeroCount < 5) {
        throw new Error("Please draw a clearer digit");
      }

      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          imageData,
          contractAddress: "0x5b73C5498c1E3b4dbA84de0F1833c4a029d90519",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const result = await response.json();

      setZkProofStatus("verified");
      setPrediction(result.prediction);
      setConfidence(result.confidence || 0.95);

      if (result.transactionHash) {
        setTransactionHash(result.transactionHash);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message);
      setZkProofStatus("failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "generating":
        return "text-yellow-500";
      case "verified":
        return "text-green-500";
      case "failed":
        return "text-red-500";
      default:
        return "text-gray-400";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "generating":
        return <Loader2 className="animate-spin h-5 w-5" />;
      case "verified":
        return <CheckCircle className="h-5 w-5" />;
      case "failed":
        return <AlertCircle className="h-5 w-5" />;
      default:
        return <Zap className="h-5 w-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            ZK-MNIST Predictor
          </h1>
          <p className="text-xl text-gray-300">
            Draw a digit and get zero-knowledge verified predictions on-chain
          </p>

          {/* Wallet Address Display */}
          {walletConnected && walletAddress && (
            <div className="mt-6 flex items-center justify-center gap-3">
              <div className="bg-white/10 backdrop-blur-lg rounded-full px-4 py-2 border border-white/20 flex items-center gap-2">
                <Wallet className="h-4 w-4 text-green-400" />
                <span className="text-sm text-gray-300">Connected:</span>
                <button
                  onClick={copyAddressToClipboard}
                  className="text-cyan-400 hover:text-cyan-300 font-mono text-sm transition-colors"
                  title="Click to copy full address"
                >
                  {formatWalletAddress(walletAddress)}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Drawing Area */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Brain className="h-6 w-6 text-cyan-400" />
              Draw Your Digit
            </h2>

            <div className="space-y-6">
              <div className="bg-black rounded-lg p-4 border-2 border-gray-600">
                <canvas
                  ref={canvasRef}
                  width={280}
                  height={280}
                  className="cursor-crosshair border rounded"
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  onTouchStart={startDrawing}
                  onTouchMove={draw}
                  onTouchEnd={stopDrawing}
                  style={{ touchAction: "none" }} // Prevent scrolling while drawing
                />
              </div>

              <div className="flex gap-4">
                <button
                  onClick={clearCanvas}
                  className="flex-1 py-3 px-6 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Clear Canvas
                </button>

                {!walletConnected ? (
                  <button
                    onClick={connectWallet}
                    className="flex-1 py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white rounded-lg transition-all transform hover:scale-105"
                  >
                    Connect Wallet
                  </button>
                ) : (
                  <button
                    onClick={submitPrediction}
                    disabled={isProcessing}
                    className="flex-1 py-3 px-6 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-all transform hover:scale-105 flex items-center justify-center gap-2"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="animate-spin h-5 w-5" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Zap className="h-5 w-5" />
                        Predict with ZK
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Results Area */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <CheckCircle className="h-6 w-6 text-green-400" />
              ZK Verification Results
            </h2>

            <div className="space-y-6">
              {/* Wallet Info */}
              {walletConnected && walletAddress && (
                <div className="bg-black/30 rounded-lg p-4">
                  <div className="text-gray-300 mb-2">Connected Wallet:</div>
                  <div className="text-sm text-cyan-400 break-all font-mono">
                    {walletAddress}
                  </div>
                  <button
                    onClick={copyAddressToClipboard}
                    className="mt-2 text-xs text-gray-400 hover:text-gray-300 transition-colors"
                  >
                    Click to copy address
                  </button>
                </div>
              )}

              {/* ZK Proof Status */}
              <div className="bg-black/30 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-gray-300">ZK Proof Status:</span>
                  <div
                    className={`flex items-center gap-2 ${getStatusColor(
                      zkProofStatus
                    )}`}
                  >
                    {getStatusIcon(zkProofStatus)}
                    <span className="capitalize">{zkProofStatus}</span>
                  </div>
                </div>
              </div>

              {/* Prediction Result */}
              {prediction !== null && (
                <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-lg p-6 border border-green-500/30">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-white mb-2">
                      {prediction}
                    </div>
                    <div className="text-lg text-gray-300">
                      Confidence: {Math.round(confidence * 100)}%
                    </div>
                  </div>
                </div>
              )}

              {/* Transaction Info */}
              {transactionHash && (
                <div className="bg-black/30 rounded-lg p-4">
                  <div className="text-gray-300 mb-2">Transaction Hash:</div>
                  <div className="text-sm text-cyan-400 break-all font-mono">
                    <a
                      href={`https://sepolia.etherscan.io/tx/${transactionHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline"
                    >
                      {transactionHash}
                    </a>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-red-400">
                    <AlertCircle className="h-5 w-5" />
                    <span>{error}</span>
                  </div>
                </div>
              )}

              {/* Info Panel */}
              <div className="bg-blue-500/20 border border-blue-500/30 rounded-lg p-4">
                <h3 className="text-blue-300 font-semibold mb-2">
                  How it works:
                </h3>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Draw a digit (0-9) on the canvas</li>
                  <li>
                    • Image is converted to 28x28 binary matrix (0s and 1s)
                  </li>
                  <li>
                    • ZK proof is generated for the neural network prediction
                  </li>
                  <li>• Proof is verified and result stored on-chain</li>
                  <li>• Your prediction is cryptographically guaranteed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MNISTPredictor;
