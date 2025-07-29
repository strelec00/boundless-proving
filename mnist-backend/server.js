// server.js - Fixed with proper data handling for RISC Zero zkVM
const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

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
  console.log("Received prediction request at:", new Date().toISOString());

  // Set timeout for the entire request to 10 minutes
  const requestTimeout = setTimeout(() => {
    if (!res.headersSent) {
      console.log("❌ Request timeout - sending 408");
      res.status(408).json({ error: "Request timeout after 10 minutes" });
    }
  }, 600000); // 10 minutes

  const cleanup = (tempImagePath) => {
    clearTimeout(requestTimeout);
    try {
      if (tempImagePath && fs.existsSync(tempImagePath)) {
        fs.unlinkSync(tempImagePath);
        console.log("🧹 Temp file cleaned up:", tempImagePath);
      }
    } catch (cleanupError) {
      console.error("Cleanup error:", cleanupError);
    }
  };

  // Enhanced process cleanup function
  const forceKillProcess = (rustProcess, reason = "timeout") => {
    if (!rustProcess || rustProcess.killed) {
      return;
    }

    console.log(`💀 Force killing Rust process (${reason})`);

    try {
      // First try SIGTERM
      rustProcess.kill("SIGTERM");

      // If still running after 5 seconds, use SIGKILL
      setTimeout(() => {
        if (!rustProcess.killed) {
          console.log("💀 Process didn't respond to SIGTERM, using SIGKILL");
          rustProcess.kill("SIGKILL");

          // Also kill by PID to be sure
          if (rustProcess.pid) {
            try {
              process.kill(rustProcess.pid, "SIGKILL");
            } catch (e) {
              console.log("Process already terminated");
            }
          }
        }
      }, 5000);
    } catch (killError) {
      console.error("Error killing process:", killError);
    }
  };

  try {
    const { imageData, contractAddress } = req.body;

    console.log("📊 Request data summary:");
    console.log(
      "- ImageData length:",
      imageData ? imageData.length : "undefined"
    );
    console.log("- Contract address:", contractAddress);

    // Enhanced validation
    if (!imageData) {
      cleanup();
      return res.status(400).json({
        error: "Missing imageData",
        details: "imageData field is required in request body",
      });
    }

    if (!contractAddress) {
      cleanup();
      return res.status(400).json({
        error: "Missing contractAddress",
        details: "contractAddress field is required in request body",
      });
    }

    // Normalize and validate image data
    let normalizedImageData;
    try {
      normalizedImageData = normalizeImageData(imageData);
    } catch (normalizeError) {
      cleanup();
      return res.status(400).json({
        error: "Invalid image data",
        details: normalizeError.message,
      });
    }

    console.log("✅ Image data validation passed");

    // Create temp directory if it doesn't exist
    const tempDir = path.join(__dirname, "temp");
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
      console.log("📁 Created temp directory:", tempDir);
    }

    // Save image to temp file
    const tempImagePath = path.join(
      tempDir,
      `image_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.json`
    );

    try {
      fs.writeFileSync(
        tempImagePath,
        JSON.stringify(normalizedImageData, null, 2)
      );
      console.log("💾 Temp image saved to:", tempImagePath);
    } catch (fileError) {
      cleanup();
      return res.status(500).json({
        error: "Failed to save temporary image file",
        details: fileError.message,
      });
    }

    // Get Rust project path
    const rustProjectPath =
      process.env.RUST_PROJECT_PATH ||
      "/Users/janstrelec/Documents/Projects/boundless-proving/zkvm";
    console.log("🦀 Starting Rust process from:", rustProjectPath);

    // Check if Rust project exists
    if (!fs.existsSync(rustProjectPath)) {
      cleanup(tempImagePath);
      return res.status(500).json({
        error: "Rust project path not found",
        path: rustProjectPath,
        suggestion:
          "Please set RUST_PROJECT_PATH environment variable or ensure the path exists",
      });
    }

    // Check if the required private key is set
    if (!process.env.PRIVATE_KEY) {
      console.warn(
        "⚠️  Warning: PRIVATE_KEY environment variable not set. Will use dry-run mode."
      );
    }

    const rustArgs = [
      "run",
      "--release", // Use release build for better performance
      "--bin",
      "app",
      "--",
      "--image-file",
      tempImagePath,
      "--mnist-predictor-address",
      contractAddress,
      "--proof-timeout",
      "480", // 8 minutes timeout for proof generation
    ];

    // Add dry-run if no private key
    if (!process.env.PRIVATE_KEY) {
      rustArgs.push("--dry-run");
    }

    console.log("🚀 Spawning Rust process with args:", rustArgs.slice(4));

    const rustProcess = spawn("cargo", rustArgs, {
      cwd: rustProjectPath,
      env: {
        ...process.env,
        RUST_LOG: "info",
        RUST_BACKTRACE: "1",
        // Add development mode for faster proof generation
      },
      stdio: "pipe",
      // Ensure the process can be killed
      detached: false,
    });

    let output = "";
    let error = "";
    let isCompleted = false;

    // INCREASED: Process timeout to 8 minutes for ZK proof generation
    const processTimeout = setTimeout(() => {
      if (!isCompleted) {
        console.log("⏰ Process timeout - killing Rust process");
        isCompleted = true;
        forceKillProcess(rustProcess, "timeout");
        cleanup(tempImagePath);
        if (!res.headersSent) {
          res.status(500).json({
            error: "ZK proof generation timeout",
            details:
              "Process exceeded 8 minute limit. This is normal for complex ZK proofs.",
            suggestion:
              "Try with RISC0_DEV_MODE=1 for faster development testing",
          });
        }
      }
    }, 480000); // 8 minutes

    // Handle process startup errors
    rustProcess.on("error", (err) => {
      if (!isCompleted) {
        console.error("❌ Failed to start Rust process:", err);
        isCompleted = true;
        clearTimeout(processTimeout);
        cleanup(tempImagePath);
        if (!res.headersSent) {
          res.status(500).json({
            error: "Failed to start Rust process",
            details: err.message,
            suggestion:
              "Make sure Rust and Cargo are installed and the project path is correct",
          });
        }
      }
    });

    // Capture stdout
    rustProcess.stdout.on("data", (data) => {
      const dataStr = data.toString();
      output += dataStr;

      // Log important messages and progress indicators
      const lines = dataStr.split("\n").filter((line) => line.trim());
      lines.forEach((line) => {
        if (
          line.includes("INFO") ||
          line.includes("ERROR") ||
          line.includes("WARN") ||
          line.includes("execution time") ||
          line.includes("Generating proof") ||
          line.includes("proof generated")
        ) {
          console.log("📤 Rust:", line.trim());
        }
      });
    });

    // Capture stderr
    rustProcess.stderr.on("data", (data) => {
      const dataStr = data.toString();
      error += dataStr;

      // Log error messages but don't spam with compilation warnings
      const lines = dataStr.split("\n").filter((line) => line.trim());
      lines.forEach((line) => {
        if (
          line.trim() &&
          !line.includes("warning:") &&
          !line.includes("Finished `")
        ) {
          console.log("🔴 Rust Error:", line.trim());
        }
      });
    });

    // Handle process completion
    rustProcess.on("close", (code) => {
      if (!isCompleted) {
        console.log(`🏁 Rust process exited with code ${code}`);
        isCompleted = true;
        clearTimeout(processTimeout);
        cleanup(tempImagePath);

        if (res.headersSent) {
          console.log("⚠️  Response already sent, skipping");
          return;
        }

        if (code === 0) {
          // Parse output to extract results
          console.log("📋 Parsing successful output...");

          try {
            const result = parseRustOutput(output);
            console.log("✅ Successfully parsed result:", result);
            res.json(result);
          } catch (parseError) {
            console.error("❌ Failed to parse output:", parseError.message);
            res.status(500).json({
              error: "Failed to parse prediction result",
              details: parseError.message,
              rawOutput: output.trim(),
              errorOutput: error.trim(),
            });
          }
        } else {
          console.error(`❌ Rust process failed with code: ${code}`);

          // Try to extract more specific error information
          let errorDetails = "Unknown error";
          if (error.includes("TypeCheckFail")) {
            errorDetails = "Data serialization error - check image data format";
          } else if (error.includes("panicked")) {
            const panicMatch = error.match(/panicked at [^:]+: (.+)/);
            if (panicMatch) {
              errorDetails = panicMatch[1];
            }
          } else if (error.includes("failed to save changes")) {
            errorDetails = "Contract compilation error - check Solidity files";
          } else if (error.includes("timeout")) {
            errorDetails =
              "ZK proof generation took too long - try with development mode";
          }

          res.status(500).json({
            error: `Rust process failed with exit code ${code}`,
            details: errorDetails,
            exitCode: code,
            errorOutput: error.trim(),
            rawOutput: output.trim(),
            suggestion:
              "Try setting RISC0_DEV_MODE=1 for faster development testing",
          });
        }
      }
    });

    // Handle process disconnect
    rustProcess.on("disconnect", () => {
      if (!isCompleted) {
        console.log("🔌 Rust process disconnected");
        isCompleted = true;
        clearTimeout(processTimeout);
        cleanup(tempImagePath);
        if (!res.headersSent) {
          res.status(500).json({
            error: "Rust process disconnected unexpectedly",
            suggestion: "Check system resources and try again",
          });
        }
      }
    });

    // Handle server shutdown - cleanup any running processes
    process.on("SIGINT", () => {
      console.log("🛑 Server shutting down, cleaning up processes...");
      if (!isCompleted && rustProcess && !rustProcess.killed) {
        forceKillProcess(rustProcess, "server shutdown");
      }
    });

    process.on("SIGTERM", () => {
      console.log("🛑 Server terminating, cleaning up processes...");
      if (!isCompleted && rustProcess && !rustProcess.killed) {
        forceKillProcess(rustProcess, "server termination");
      }
    });
  } catch (err) {
    console.error("❌ API Error:", err);
    cleanup();
    if (!res.headersSent) {
      res.status(500).json({
        error: "Internal server error",
        details: err.message,
        stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
      });
    }
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
