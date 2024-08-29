const express = require('express');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const app = express();

// Middlewares
app.use(cors());
app.use(express.json()); // for parsing application/json

// TensorFlow model
let model;

// Load the TensorFlow model
const loadModel = async () => {
  model = await tf.loadLayersModel('file://C:/path_to_your_model/my_model/model.json');
  console.log("Model loaded successfully.");
};

loadModel();

// Prediction endpoint
app.post('/predict', async (req, res) => {
  if (!model) {
    return res.status(500).json({ error: 'Model not loaded yet' });
  }

  try {
    const inputData = tf.tensor(req.body.data);
    const prediction = model.predict(inputData);
    prediction.array().then(result => {
      res.json({ prediction: result });
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Set the port
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});