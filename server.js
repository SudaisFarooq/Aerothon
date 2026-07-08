const express = require('express');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const DB_FILE = path.join(__dirname, 'database.json');

// Middleware
app.use(express.json());

// Catch and handle invalid JSON format requests gracefully
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({ error: 'Invalid JSON payload format.' });
  }
  next();
});

app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Ensure database file exists
if (!fs.existsSync(DB_FILE)) {
  fs.writeFileSync(DB_FILE, JSON.stringify([], null, 2));
}

// Authentication Helper
function isAuthorized(req) {
  const authHeader = req.headers['authorization'] || '';
  const expectedPassword = process.env.DASHBOARD_PASSWORD || 'admin123';
  return authHeader === expectedPassword;
}

// API Routes

// 1. Submit Feedback (POST /api/feedbacks)
app.post('/api/feedbacks', (req, res) => {
  const { email, message } = req.body;

  if (!email || !message) {
    return res.status(400).json({ error: 'Email and message are required.' });
  }

  try {
    const data = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    
    const newFeedback = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
      email: email.trim(),
      message: message.trim(),
      timestamp: new Date().toISOString()
    };

    data.unshift(newFeedback); // Add to the beginning so it shows first on the dashboard
    fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));

    res.status(201).json({ success: true, feedback: newFeedback });
  } catch (error) {
    console.error('Error writing feedback:', error);
    res.status(500).json({ error: 'Failed to save feedback.' });
  }
});

// 2. Get Feedbacks (GET /api/feedbacks)
app.get('/api/feedbacks', (req, res) => {
  if (!isAuthorized(req)) {
    return res.status(401).json({ error: 'Unauthorized. Invalid passcode.' });
  }

  try {
    const data = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    res.json(data);
  } catch (error) {
    console.error('Error reading feedbacks:', error);
    res.status(500).json({ error: 'Failed to read feedback database.' });
  }
});

// 3. Delete Feedback (DELETE /api/feedbacks/:id)
app.delete('/api/feedbacks/:id', (req, res) => {
  if (!isAuthorized(req)) {
    return res.status(401).json({ error: 'Unauthorized.' });
  }

  const { id } = req.params;

  try {
    let data = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    const initialLength = data.length;
    data = data.filter(item => item.id !== id);

    if (data.length === initialLength) {
      return res.status(404).json({ error: 'Feedback not found.' });
    }

    fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));
    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting feedback:', error);
    res.status(500).json({ error: 'Failed to delete feedback.' });
  }
});

// Serve frontend for /dashboard URL
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

// Start Server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`Feedback app is running locally!`);
  console.log(`Frontend Form: http://localhost:${PORT}`);
  console.log(`Dashboard:     http://localhost:${PORT}/dashboard`);
  console.log(`Default Passcode: admin123`);
  console.log(`==================================================`);
});
