# Interactive Handwritten Digit Recognition (MLP)

A fully self-contained, real-time handwritten digit recognition web application. The machine learning model is an MLP (Multi-Layer Perceptron) neural network trained in Python on the MNIST dataset and runs entirely client-side in the browser using raw JavaScript matrix math (zero libraries for inference!).

## Live Demo
Deployed on GitHub Pages: [Live Application](https://om-mane-coder.github.io/handwritten-digit-recognition-mlp/)

## Key Features
* **Interactive Canvas:** Draw digits directly in the browser with mouse or touch gestures.
* **Real-time Prediction:** Automatic feedforward neural network execution as you draw (throttled to keep performance at 60 FPS).
* **Pure JavaScript Inference:** No heavy frameworks (TensorFlow.js/ONNX) loaded on run-time. The feedforward dot product and ReLU/Softmax calculations are coded in raw JS.
* **Confidence Breakdown:** Beautiful visual charts showing the model's confidence scores for digits 0-9.

## Model Architecture
* **Input Layer:** 784 Neurons (28x28 grayscale canvas output normalized between 0.0 and 1.0).
* **Hidden Layer 1:** 128 Neurons with Rectified Linear Unit (ReLU) activation.
* **Hidden Layer 2:** 64 Neurons with Rectified Linear Unit (ReLU) activation.
* **Output Layer:** 10 Neurons with Softmax activation (digits 0-9).
* **Accuracy:** 96.48% on the MNIST testing partition.

## Development Stack
* **Training Pipeline:** Python, Scikit-learn, Numpy
* **Web Frontend:** Vanilla HTML5 Canvas, CSS Variables, Glassmorphic styling, Vanilla JavaScript (ES6)

## Getting Started
Simply double-click `index.html` to run locally, or deploy the folder directly as a static site to Vercel or GitHub Pages.
