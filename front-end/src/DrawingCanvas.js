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
  };

  const downloadMatrix = () => {
    // Flatten the 28x28 matrix into a single array of 784 elements
    const flatMatrix = matrix.flat();

    // Format as Rust array with 28 elements per line
    let rustFormat = "pub const SAMPLE: [i32; 784] = [\n";
    for (let i = 0; i < flatMatrix.length; i += 28) {
      const row = flatMatrix.slice(i, i + 28);
      rustFormat += "    " + row.join(", ") + ",\n";
    }
    rustFormat += "];";

    const blob = new Blob([rustFormat], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "mnist_image.rs";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const togglePreview = () => {
    setShowPreview(!showPreview);
  };

  return (
    <div className="flex flex-col items-center p-5 bg-gray-100 min-h-screen">
      <div className="bg-white p-5 rounded-lg shadow-lg text-center max-w-2xl">
        <h1 className="text-2xl font-bold text-gray-800 mb-5">
          28×28 Binary Drawing Canvas
        </h1>

        <div className="bg-blue-50 p-4 rounded-lg mb-5 text-left">
          <strong className="text-blue-800">How to Use:</strong>
          <ol className="list-decimal list-inside mt-2 text-sm text-gray-700 space-y-1">
            <li>Click and drag on the canvas below to draw</li>
            <li>
              Your drawing automatically converts to a 28×28 binary matrix
            </li>
            <li>Click "Toggle Preview" to see the 0s and 1s</li>
            <li>Click "Download Rust File" to save for your CLI tool</li>
            <li>Use "Clear Canvas" to start over</li>
          </ol>
        </div>

        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="border-2 border-gray-800 cursor-crosshair bg-white my-5"
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
            onClick={downloadMatrix}
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Download Rust File
          </button>
          <button
            onClick={togglePreview}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            {showPreview ? "Hide Preview" : "Show Preview"}
          </button>
        </div>

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
          <strong className="text-yellow-800">Technical Details:</strong>
          <ul className="list-disc list-inside mt-2 text-gray-700 space-y-1">
            <li>Canvas: 280×280 pixels (scaled from 28×28)</li>
            <li>Output: Rust const array with 784 i32 values</li>
            <li>File: mnist_image.rs compatible with your CLI</li>
            <li>Perfect for Rust CLI and ML projects</li>
          </ul>
        </div>

        <div className="bg-green-50 p-4 rounded-lg mt-5 text-left text-sm">
          <strong className="text-green-800">CLI Usage:</strong>
          <div className="mt-2 font-mono text-xs bg-gray-800 text-green-300 p-2 rounded">
            RUST_LOG=info cargo run --bin app -- --image-file ./mnist_image.rs
            [other-args]
          </div>
        </div>
      </div>
    </div>
  );
};

export default DrawingCanvas;
