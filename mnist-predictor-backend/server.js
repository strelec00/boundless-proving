const express = require('express');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3001;

// Relativna putanja do glavnog Rust projekta (jedan nivo gore od mnist_backend)
const RUST_PROJECT_PATH = path.resolve(__dirname, '..');  // Ide jedan direktorij gore

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'Server is running!', 
    timestamp: new Date().toISOString(),
    rustProjectPath: RUST_PROJECT_PATH,
    absolutePath: path.resolve(RUST_PROJECT_PATH),
    currentDir: __dirname
  });
});

// Main prediction endpoint
app.post('/api/predict', async (req, res) => {
  console.log('Received prediction request...');
  console.log('This may take several minutes for proof generation and blockchain verification...');
  
  try {
    const { rustFormat, matrix } = req.body;
    
    if (!rustFormat || !matrix) {
      return res.status(400).json({ 
        success: false, 
        error: 'Missing rustFormat or matrix data' 
      });
    }

    // Save canvas.rs in the main Rust project directory (parent of mnist_backend)
    const filePath = path.join(RUST_PROJECT_PATH, 'canvas.rs');
    fs.writeFileSync(filePath, rustFormat);
    console.log('Canvas file saved to:', filePath);
    console.log('Absolute path:', path.resolve(filePath));
    
    // Run the CLI command from your main Rust project directory
    const cmd = `RUST_LOG=info MNIST_PREDICTOR_ADDRESS="0x9634d65c6C38877E6ca9730c1bD86762695C1cC3" cargo run --bin app -- --image-file ./canvas.rs`;
    
    console.log('Running command from:', RUST_PROJECT_PATH);
    console.log('Command:', cmd);
    console.log('Starting process (this will take time for proof generation)...');
    
    const childProcess = exec(cmd, { 
      cwd: RUST_PROJECT_PATH,
      timeout: 600000, // 10 minutes timeout for proof generation + blockchain
      maxBuffer: 10 * 1024 * 1024, // Increased buffer for more output
      env: {
        ...process.env,
        RUST_LOG: 'info',
        MNIST_PREDICTOR_ADDRESS: '0x9634d65c6C38877E6ca9730c1bD86762695C1cC3'
      }
    });

    // Collect all output
    let fullStdout = '';
    let fullStderr = '';

    // Stream stdout in real-time to console
    childProcess.stdout.on('data', (data) => {
      const output = data.toString();
      fullStdout += output;
      console.log('[RUST STDOUT]:', output);
    });

    // Stream stderr in real-time to console
    childProcess.stderr.on('data', (data) => {
      const output = data.toString();
      fullStderr += output;
      console.log('[RUST STDERR]:', output);
    });

    // Handle process completion
    childProcess.on('exit', (code, signal) => {
      console.log(`Process exited with code ${code} and signal ${signal}`);
    });

    // Wait for process to complete
    childProcess.on('close', (code, signal) => {
      console.log('=== FINAL OUTPUT ===');
      console.log('Exit code:', code);
      console.log('Signal:', signal);
      
      // Extract prediction from output (check both stdout and stderr)
      const predictionMatch = fullStdout.match(/Predicted digit: (\d+)/i) || 
                            fullStderr.match(/Predicted digit: (\d+)/i);
      const prediction = predictionMatch ? parseInt(predictionMatch[1]) : null;
      
      // Check for blockchain transaction
      const txMatch = fullStdout.match(/Tx hash: (0x[a-fA-F0-9]+)/i) || 
                     fullStderr.match(/Tx hash: (0x[a-fA-F0-9]+)/i);
      const txHash = txMatch ? txMatch[1] : null;
      
      // Check if proof was generated
      const proofGenerated = fullStdout.includes('Generating proof') || 
                           fullStdout.includes('proof generated') ||
                           fullStderr.includes('Generating proof') ||
                           fullStderr.includes('proof generated');
      
      // Check if blockchain verification succeeded
      const blockchainVerified = fullStdout.includes('Transaction confirmed') || 
                                fullStdout.includes('Tx confirmed') ||
                                fullStdout.includes('successfully verified');
      
      // Determine success based on what completed
      let success = false;
      let message = '';
      
      if (prediction !== null) {
        success = true;
        if (blockchainVerified && txHash) {
          message = `Prediction completed and verified on blockchain! Tx: ${txHash}`;
        } else if (proofGenerated) {
          message = `Prediction completed with proof generation! Predicted: ${prediction}`;
        } else {
          message = `Prediction completed! Predicted digit: ${prediction}`;
        }
      }
      
      // Send response
      res.json({ 
        success,
        prediction,
        txHash,
        proofGenerated,
        blockchainVerified,
        message: message || 'Process completed',
        exitCode: code,
        signal,
        stdout: fullStdout.substring(0, 5000), // Limit response size
        stderr: fullStderr.substring(0, 5000),
        canvasPath: filePath
      });
    });

    // Handle errors
    childProcess.on('error', (error) => {
      console.error('Process error:', error);
      res.status(500).json({ 
        success: false, 
        error: `Process error: ${error.message}` 
      });
    });
    
  } catch (err) {
    console.error('Server error:', err);
    res.status(500).json({ 
      success: false, 
      error: `Server error: ${err.message}` 
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 MNIST Predictor Backend running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔮 Prediction API: POST http://localhost:${PORT}/api/predict`);
  console.log(`📁 Rust project path: ${RUST_PROJECT_PATH}`);
  console.log(`📁 Absolute path: ${path.resolve(RUST_PROJECT_PATH)}`);
  console.log(`📁 Current directory: ${__dirname}`);
  console.log(`⏱️  Timeout set to 10 minutes for proof generation + blockchain verification`);
});

process.on('SIGINT', () => {
  console.log('\n👋 Shutting down server...');
  process.exit(0);
});