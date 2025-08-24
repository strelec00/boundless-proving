import React, { useState, useRef, useEffect } from "react";

const DrawingCanvas = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [matrix, setMatrix] = useState(() =>
    Array(28)
      .fill()
      .map(() => Array(28).fill(0))
  );
  const [showPreview, setShowPreview] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const scale = 10; // Canvas is 280x280 but represents 28x28

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Initialize canvas
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
  }, []);

  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Draw on canvas
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fill();

    // Update matrix
    const matrixX = Math.floor(x / scale);
    const matrixY = Math.floor(y / scale);

    if (matrixX >= 0 && matrixX < 28 && matrixY >= 0 && matrixY < 28) {
      setMatrix((prevMatrix) => {
        const newMatrix = prevMatrix.map((row) => [...row]);
        newMatrix[matrixY][matrixX] = 1;

        // Fill nearby pixels for better coverage
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const newX = matrixX + dx;
            const newY = matrixY + dy;
            if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
              newMatrix[newY][newX] = 1;
            }
          }
        }
        return newMatrix;
      });
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const handleTouch = (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
      e.type === "touchstart" ? "mousedown" : "mousemove",
      {
        clientX: touch.clientX,
        clientY: touch.clientY,
      }
    );
    canvasRef.current.dispatchEvent(mouseEvent);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setMatrix(
      Array(28)
        .fill()
        .map(() => Array(28).fill(0))
    );
    setPrediction(null);
    setError(null);
  };

  const saveCanvasFile = async () => {
    // Flatten the 28x28 matrix into a single array of 784 elements
    const flatMatrix = matrix.flat();

    // Format as Rust array
    let rustFormat = "pub const SAMPLE: [i32; 784] = [\n";
    for (let i = 0; i < flatMatrix.length; i += 28) {
      const row = flatMatrix.slice(i, i + 28);
      rustFormat += "    " + row.join(", ") + ",\n";
    }
    rustFormat += "];";

    // Create and download the file
    const blob = new Blob([rustFormat], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "canvas.rs";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    return "canvas.rs";
  };

  const predictDigit = async () => {
    try {
      setIsPredicting(true);
      setError(null);
      setPrediction(null);

      // Flatten the 28x28 matrix into a single array of 784 elements
      const flatMatrix = matrix.flat();

      // Send the matrix data to your backend API
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          matrix: flatMatrix,
          // You can also send the full Rust format if needed
          rustFormat: generateRustFormat(flatMatrix)
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setPrediction(result.prediction);
      } else {
        throw new Error(result.error || 'Prediction failed');
      }

    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const generateRustFormat = (flatMatrix) => {
    let rustFormat = "pub const SAMPLE: [i32; 784] = [\n";
    for (let i = 0; i < flatMatrix.length; i += 28) {
      const row = flatMatrix.slice(i, i + 28);
      rustFormat += "    " + row.join(", ") + ",\n";
    }
    rustFormat += "];";
    return rustFormat;
  };

  const downloadMatrix = () => {
    saveCanvasFile();
  };

  const togglePreview = () => {
    setShowPreview(!showPreview);
  };

  const hasDrawing = matrix.some(row => row.some(pixel => pixel === 1));

  return (
    <div className="flex flex-col items-center p-5 bg-gray-100 min-h-screen">
      <div className="bg-white p-5 rounded-lg shadow-lg text-center max-w-2xl">
        <h1 className="text-2xl font-bold text-gray-800 mb-5">
          MNIST Digit Predictor
        </h1>

        <div className="bg-blue-50 p-4 rounded-lg mb-5 text-left">
          <strong className="text-blue-800">How to Use:</strong>
          <ol className="list-decimal list-inside mt-2 text-sm text-gray-700 space-y-1">
            <li>Draw a digit (0-9) on the canvas below</li>
            <li>Click "Predict" to save the file and get CLI command</li>
            <li>Run the CLI command in your terminal</li>
            <li>The prediction will be verified on blockchain</li>
          </ol>
        </div>

        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="border-2 border-gray-800 cursor-crosshair bg-white my-5 rounded-lg"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={handleTouch}
          onTouchMove={handleTouch}
          onTouchEnd={stopDrawing}
        />

        <div className="flex gap-3 justify-center flex-wrap my-5">
          <button
            onClick={clearCanvas}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Clear Canvas
          </button>
          
          <button
            onClick={predictDigit}
            disabled={!hasDrawing || isPredicting}
            className={`px-6 py-2 rounded-lg transition-colors font-medium ${
              !hasDrawing || isPredicting
                ? "bg-gray-400 cursor-not-allowed text-gray-600"
                : "bg-purple-600 hover:bg-purple-700 text-white"
            }`}
          >
            {isPredicting ? "Processing..." : "🔮 Predict"}
          </button>

          <button
            onClick={downloadMatrix}
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Download File
          </button>
          
          <button
            onClick={togglePreview}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            {showPreview ? "Hide Preview" : "Show Matrix"}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg my-3">
            <strong>Error:</strong> {error}
          </div>
        )}

        {prediction !== null && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg my-3">
            <strong>Prediction:</strong> The digit is <span className="text-2xl font-bold">{prediction}</span>
          </div>
        )}

        {showPreview && (
          <div className="bg-gray-50 p-4 rounded-lg my-5 font-mono text-xs leading-none max-h-48 overflow-y-auto">
            <strong className="text-gray-800 font-sans text-sm">
              Matrix Preview (28×28):
            </strong>
            <div className="mt-2 text-left">
              {matrix.map((row, i) => (
                <div key={i} className="mb-1">
                  {row.join(" ")}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="bg-yellow-50 p-4 rounded-lg mt-5 text-left text-sm">
          <strong className="text-yellow-800">CLI Command:</strong>
          <div className="mt-2 font-mono text-xs bg-gray-800 text-green-300 p-3 rounded overflow-x-auto">
            RUST_LOG=info MNIST_PREDICTOR_ADDRESS="0x9634d65c6C38877E6ca9730c1bD86762695C1cC3" cargo run --bin app -- --image-file ./canvas.rs
          </div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg mt-5 text-left text-sm">
          <strong className="text-green-800">Technical Details:</strong>
          <ul className="list-disc list-inside mt-2 text-gray-700 space-y-1">
            <li>Canvas: 280×280 pixels → 28×28 binary matrix</li>
            <li>Output: canvas.rs with Rust const array</li>
            <li>Blockchain verification via RISC Zero</li>
            <li>Contract: 0x9634d65c6C38877E6ca9730c1bD86762695C1cC3</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DrawingCanvas;