// server.js - Fixed with proper data handling for RISC Zero zkVM
const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const PROOF_TIMEOUT = parseInt(process.env.PROOF_TIMEOUT || "600") * 1000; // Default 10 minutes
const app = express();
const PORT = process.env.PORT || 3001;

// Enhanced CORS configuration
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "http://127.0.0.1:3000",
      "http://localhost:3001",
      "http://127.0.0.1:3001",
    ],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "Accept"],
    credentials: true,
    optionsSuccessStatus: 200,
  })
);

// Add request logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

app.use(express.json({ limit: "10mb" }));

// Add response headers middleware
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", req.headers.origin || "*");
  res.header("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS");
  res.header(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, Content-Length, X-Requested-With, Accept"
  );
  res.header("Access-Control-Allow-Credentials", "true");

  if (req.method === "OPTIONS") {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Helper function to normalize and validate image data
function normalizeImageData(imageData) {
  console.log("🔧 Normalizing image data...");

  if (!Array.isArray(imageData)) {
    throw new Error("imageData must be an array");
  }

  if (imageData.length !== 784) {
    throw new Error(
      `Invalid image data length. Expected 784, got ${imageData.length}`
    );
  }

  // Check data range and normalize
  const maxValue = Math.max(...imageData);
  const minValue = Math.min(...imageData);

  console.log(`📊 Original data range: ${minValue} - ${maxValue}`);

  let normalizedData;

  if (maxValue > 1) {
    // Normalize from 0-255 to 0-1 (binary)
    normalizedData = imageData.map((pixel) => {
      const normalized = Math.max(0, Math.min(255, Math.round(pixel)));
      return normalized > 128 ? 1 : 0; // Binary threshold
    });
    console.log("🔄 Normalized 0-255 range to binary (0-1)");
  } else {
    // Already in 0-1 range, ensure integers
    normalizedData = imageData.map((pixel) => {
      return Math.max(0, Math.min(1, Math.round(pixel)));
    });
    console.log("✅ Data already in binary range");
  }

  // Validate normalized data
  const invalidValues = normalizedData.filter((val) => val !== 0 && val !== 1);
  if (invalidValues.length > 0) {
    throw new Error(
      `Invalid pixel values after normalization: ${invalidValues.slice(0, 5)}`
    );
  }

  const nonZeroCount = normalizedData.filter((val) => val === 1).length;
  console.log(
    `📈 Non-zero pixels: ${nonZeroCount}/784 (${(
      (nonZeroCount / 784) *
      100
    ).toFixed(1)}%)`
  );

  if (nonZeroCount === 0) {
    throw new Error("Image appears to be completely empty (all pixels are 0)");
  }

  if (nonZeroCount < 10) {
    console.warn(
      `⚠️  Warning: Image has very few non-zero pixels (${nonZeroCount})`
    );
  }

  return normalizedData;
}

// Endpoint for MNIST prediction
// Replace the process handling section in your server.js with this improved version

app.post("/api/predict", async (req, res) => {
  console.log("=== PREDICTION REQUEST START ===");

  // Set appropriate timeout for the request
  req.setTimeout(PROOF_TIMEOUT + 60000); // Add 1 minute buffer
  res.setTimeout(PROOF_TIMEOUT + 60000);

  const cleanup = (tempImagePath) => {
    try {
      if (tempImagePath && fs.existsSync(tempImagePath)) {
        fs.unlinkSync(tempImagePath);
        console.log("🧹 Temp file cleaned up:", tempImagePath);
      }
    } catch (cleanupError) {
      console.error("Cleanup error:", cleanupError);
    }
  };

  try {
    const { imageData, contractAddress } = req.body;

    // Validation
    if (!imageData || !Array.isArray(imageData) || imageData.length !== 784) {
      return res.status(400).json({
        error: "Invalid image data",
        details: "imageData must be an array of 784 values",
      });
    }

    if (!contractAddress) {
      return res.status(400).json({
        error: "Missing contractAddress",
      });
    }

    // Normalize image data
    const normalizedImageData = normalizeImageData(imageData);

    // Save to temp file
    const tempDir = path.join(__dirname, "temp");
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    const tempImagePath = path.join(tempDir, `image_${Date.now()}.json`);
    fs.writeFileSync(tempImagePath, JSON.stringify(normalizedImageData));

    // Spawn Rust process with proper environment
    const rustArgs = [
      "run",
      "--release",
      "--bin",
      "app",
      "--",
      "--image-file",
      tempImagePath,
      "--mnist-predictor-address",
      contractAddress,
      "--proof-timeout",
      Math.floor(PROOF_TIMEOUT / 1000).toString(),
      "--offchain", // Add this for more reliable proving
    ];
    if (!process.env.PRIVATE_KEY) {
      rustArgs.push("--dry-run");
    }

    // In server.js, update the spawn call to properly pass environment variables
    const rustProcess = spawn("cargo", rustArgs, {
      cwd:
        process.env.RUST_PROJECT_PATH ||
        "/Users/janstrelec/Documents/Projects/boundless-proving/zkvm",
      env: {
        ...process.env,
        RUST_LOG: "info",
        RISC0_DEV_MODE: process.env.RISC0_DEV_MODE || "0", // Set to "0" for production
        // Make sure these are passed through:
        PRIVATE_KEY: process.env.PRIVATE_KEY,
        RPC_URL:
          process.env.RPC_URL || "https://ethereum-sepolia-rpc.publicnode.com",
        MNIST_PREDICTOR_ADDRESS: contractAddress, // This comes from the request
      },
      stdio: "pipe",
      detached: false,
    });

    // Handle process output
    let output = "";
    let error = "";

    rustProcess.stdout.on("data", (data) => {
      output += data.toString();
      console.log("Rust:", data.toString().trim());
    });

    rustProcess.stderr.on("data", (data) => {
      error += data.toString();
    });

    // Wait for process completion
    const result = await new Promise((resolve, reject) => {
      rustProcess.on("close", (code) => {
        cleanup(tempImagePath);

        if (code === 0) {
          try {
            const parsedResult = parseRustOutput(output);
            resolve(parsedResult);
          } catch (parseError) {
            reject(new Error(`Failed to parse output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Process failed with code ${code}: ${error}`));
        }
      });

      rustProcess.on("error", (err) => {
        cleanup(tempImagePath);
        reject(err);
      });

      // Add timeout
      setTimeout(() => {
        rustProcess.kill("SIGTERM");
        cleanup(tempImagePath);
        reject(new Error("Process timeout"));
      }, PROOF_TIMEOUT);
    });

    res.json(result);
  } catch (err) {
    console.error("API Error:", err);
    res.status(500).json({
      error: err.message,
      suggestion: process.env.RISC0_DEV_MODE
        ? null
        : "Try setting RISC0_DEV_MODE=1 for faster development",
    });
  }
});

// Helper function to kill zombie processes on startup
function cleanupZombieProcesses() {
  const { exec } = require("child_process");

  // Kill any remaining cargo processes related to our project
  exec('pkill -f "cargo run.*mnist"', (error, stdout, stderr) => {
    if (stdout || stderr) {
      console.log("🧹 Cleaned up zombie processes");
    }
  });
}

// Call cleanup on server start
cleanupZombieProcesses();

// Helper function to parse Rust output
function parseRustOutput(output) {
  console.log("🔍 Parsing Rust output...");

  // Look for prediction in the summary section or logs
  const predictionPatterns = [
    /Predicted [Dd]igit:?\s*(\d+)/,
    /🎯.*?[Pp]redicted.*?[Dd]igit:?\s*(\d+)/,
    /prediction:\s*(\d+)/i,
  ];

  let prediction = null;
  for (const pattern of predictionPatterns) {
    const match = output.match(pattern);
    if (match) {
      prediction = parseInt(match[1]);
      break;
    }
  }

  if (prediction === null) {
    throw new Error("Could not find prediction value in output");
  }

  // Look for confidence
  const confidencePatterns = [
    /Confidence:?\s*([\d.]+)%?/i,
    /📈.*?Confidence:?\s*([\d.]+)%?/,
  ];

  let confidence = 0.8; // default
  for (const pattern of confidencePatterns) {
    const match = output.match(pattern);
    if (match) {
      confidence = parseFloat(match[1]);
      if (confidence > 1) confidence = confidence / 100; // Convert percentage to decimal
      break;
    }
  }

  // Look for transaction hash
  const txHashPatterns = [
    /Transaction hash:?\s*(0x[a-fA-F0-9]+)/,
    /TX Hash:?\s*(0x[a-fA-F0-9]+)/,
    /📄.*?TX Hash:?\s*(0x[a-fA-F0-9]+)/,
  ];

  let transactionHash = null;
  for (const pattern of txHashPatterns) {
    const match = output.match(pattern);
    if (match) {
      transactionHash = match[1];
      break;
    }
  }

  // Look for gas used
  const gasPatterns = [/Gas [Uu]sed:?\s*(\d+)/, /⛽.*?Gas [Uu]sed:?\s*(\d+)/];

  let gasUsed = null;
  for (const pattern of gasPatterns) {
    const match = output.match(pattern);
    if (match) {
      gasUsed = parseInt(match[1]);
      break;
    }
  }

  // Look for proof generation time
  const timePatterns = [
    /proof generated in ([\d.]+)s/i,
    /Proof Generation:?\s*([\d.]+)s/i,
  ];

  let proofTime = null;
  for (const pattern of timePatterns) {
    const match = output.match(pattern);
    if (match) {
      proofTime = parseFloat(match[1]) * 1000; // Convert to milliseconds
      break;
    }
  }

  const result = {
    prediction,
    confidence,
    transactionHash,
    gasUsed,
    proofGenerationTimeMs: proofTime,
    rawOutput: output.trim(),
    timestamp: new Date().toISOString(),
  };

  // Add dry run indicator if no transaction hash
  if (!transactionHash && output.includes("dry run")) {
    result.dryRun = true;
    result.note = "Dry run mode - no blockchain transaction submitted";
  }

  return result;
}

// Health check endpoint
app.get("/api/health", (req, res) => {
  console.log("🏥 Health check requested");

  const rustProjectPath =
    process.env.RUST_PROJECT_PATH ||
    "/Users/janstrelec/Documents/Projects/boundless-proving/zkvm";

  const health = {
    status: "OK",
    timestamp: new Date().toISOString(),
    pid: process.pid,
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    rustPath: rustProjectPath,
    rustPathExists: fs.existsSync(rustProjectPath),
    hasPrivateKey: !!process.env.PRIVATE_KEY,
    nodeVersion: process.version,
    platform: process.platform,
  };

  console.log("📊 Health status:", {
    ...health,
    memory: `${Math.round(health.memory.heapUsed / 1024 / 1024)}MB used`,
  });

  res.json(health);
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("💥 Unhandled error:", err);
  if (!res.headersSent) {
    res.status(500).json({
      error: "Internal server error",
      details:
        process.env.NODE_ENV === "development"
          ? err.message
          : "Something went wrong",
      timestamp: new Date().toISOString(),
    });
  }
});

// Create temp directory on startup
const tempDir = path.join(__dirname, "temp");
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
  console.log("📁 Created temp directory:", tempDir);
}

// Cleanup old temp files on startup
try {
  const files = fs.readdirSync(tempDir);
  const now = Date.now();
  let cleanedCount = 0;

  files.forEach((file) => {
    const filePath = path.join(tempDir, file);
    const stats = fs.statSync(filePath);
    const ageHours = (now - stats.mtime.getTime()) / (1000 * 60 * 60);

    if (ageHours > 1) {
      // Remove files older than 1 hour
      fs.unlinkSync(filePath);
      cleanedCount++;
    }
  });

  if (cleanedCount > 0) {
    console.log(`🧹 Cleaned up ${cleanedCount} old temp files`);
  }
} catch (cleanupError) {
  console.warn("⚠️  Could not cleanup old temp files:", cleanupError.message);
}

app.listen(PORT, () => {
  console.log(`\n🚀 === MNIST ZK PREDICTION SERVER STARTED ===`);
  console.log(`📡 Server running on port ${PORT}`);
  console.log(
    `🦀 Rust project path: ${
      process.env.RUST_PROJECT_PATH ||
      "/Users/janstrelec/Documents/Projects/boundless-proving/zkvm"
    }`
  );
  console.log(`📁 Temp directory: ${tempDir}`);
  console.log(`🔑 Private key configured: ${!!process.env.PRIVATE_KEY}`);
  console.log(`🌐 Node.js version: ${process.version}`);
  console.log(`💻 Platform: ${process.platform}`);
  console.log(`===============================================\n`);
});

module.exports = app;
