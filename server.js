const express = require('express');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Express middleware setup
app.use(express.json());

// Handle invalid JSON requests gracefully
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({ error: 'Malformed JSON payload.' });
  }
  next();
});

app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Subprocess mapping to invoke PyTorch predictions
app.post('/api/predict', (req, res) => {
  const {
    Cycle, Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min, 
    FuelFlow_kg_s, P2_Pa, T2_K, P3_Pa, T3_K, P4_Pa, T4_K
  } = req.body;

  const inputs = [
    Cycle, Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min, 
    FuelFlow_kg_s, P2_Pa, T2_K, P3_Pa, T3_K, P4_Pa, T4_K
  ];

  // Verify all 13 features exist and are numeric
  if (inputs.some(val => val === undefined || val === null || isNaN(Number(val)))) {
    return res.status(400).json({ error: 'Missing or non-numeric telemetry inputs. 13 parameters are required.' });
  }

  // Format arguments to invoke predict.py as a separate process
  const args = inputs.map(val => Number(val)).join(' ');
  const cmd = `python predict.py ${args}`;

  exec(cmd, (error, stdout, stderr) => {
    if (error) {
      console.error(`Subprocess error: ${error}`);
      console.error(`Stderr output: ${stderr}`);
      return res.status(500).json({ error: 'Inference model execution failed.', details: stderr });
    }
    
    try {
      // Find the JSON line in output stream
      const lines = stdout.trim().split('\n');
      const jsonLine = lines.find(l => l.trim().startsWith('{') && l.trim().endsWith('}'));
      if (!jsonLine) {
        throw new Error('No JSON predictions found in python response stream.');
      }
      const pred = JSON.parse(jsonLine.trim());
      res.json(pred);
    } catch (parseError) {
      console.error(`Parsing error on stdout: "${stdout}"`);
      res.status(500).json({ error: 'Failed to parse model predictions output.', details: stdout });
    }
  });
});

// Launch server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`Aerothona Turbojet Diagnostic Digital Twin active!`);
  console.log(`Local Access URL: http://localhost:${PORT}`);
  console.log(`==================================================`);
});
