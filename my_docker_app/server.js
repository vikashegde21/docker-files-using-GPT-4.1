// Simple Node.js app with MongoDB
const express = require('express');
const { MongoClient } = require('mongodb');

const app = express();
const PORT = process.env.PORT || 3000;
const MONGO_URL = process.env.MONGO_URL || 'mongodb://mongo:27017';
const DB_NAME = process.env.MONGO_DB || 'testdb';

let db;

// Health endpoint
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Root endpoint
app.get('/', async (req, res) => {
  try {
    const collection = db.collection('visits');
    await collection.insertOne({ timestamp: new Date() });
    const count = await collection.countDocuments();
    res.send(`<h1>Hello from Node.js + MongoDB!</h1><p>Visits: ${count}</p>`);
  } catch (err) {
    res.status(500).send('Database error: ' + err.message);
  }
});

async function start() {
  try {
    const client = new MongoClient(MONGO_URL);
    await client.connect();
    db = client.db(DB_NAME);
    console.log('Connected to MongoDB');
    app.listen(PORT, () => {
      console.log(`App listening on port ${PORT}`);
    });
  } catch (err) {
    console.error('Failed to connect to MongoDB:', err);
    process.exit(1);
  }
}

start();
