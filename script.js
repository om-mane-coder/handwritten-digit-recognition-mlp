// Neural Network Weights & Biases
let model = null;

// UI Elements
const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const autoPredict = document.getElementById('autoPredict');
const predictedDigitText = document.getElementById('predictedDigit');
const modelStatus = document.getElementById('modelStatus');
const confidenceBarsContainer = document.getElementById('confidenceBars');

// Drawing state
let isDrawing = false;
let hasDrawn = false;

// Initialize drawing canvas properties
ctx.strokeStyle = '#FFFFFF';
ctx.fillStyle = '#000000';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Clear canvas to solid black
function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    predictedDigitText.innerText = '-';
    hasDrawn = false;
    updateConfidenceBars(new Array(10).fill(0));
}

// Set up drawing event listeners
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function startDrawing(e) {
    isDrawing = true;
    hasDrawn = true;
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    e.preventDefault();
}

function draw(e) {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    e.preventDefault();
    
    if (autoPredict.checked && model) {
        debouncedPredict();
    }
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// Touch support for mobile devices
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', () => {
    if (model) performInference();
});

// Load the trained model weights
async function loadModel() {
    try {
        const response = await fetch('mnist_model.json');
        if (!response.ok) {
            throw new Error(`Failed to load weights: ${response.statusText}`);
        }
        model = await response.json();
        modelStatus.innerText = 'Model Ready';
        modelStatus.classList.add('ready');
    } catch (err) {
        console.error(err);
        modelStatus.innerText = 'Error Loading Model';
        modelStatus.classList.add('error');
    }
}

// Generate the 0-9 bars layout
function initConfidenceBars() {
    confidenceBarsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.id = `bar-row-${i}`;
        row.innerHTML = `
            <span class="bar-label">${i}</span>
            <div class="bar-track">
                <div class="bar-fill" id="bar-fill-${i}"></div>
            </div>
            <span class="bar-val" id="bar-val-${i}">0.0%</span>
        `;
        confidenceBarsContainer.appendChild(row);
    }
}

function updateConfidenceBars(probs) {
    const maxValIdx = probs.indexOf(Math.max(...probs));
    
    for (let i = 0; i < 10; i++) {
        const row = document.getElementById(`bar-row-${i}`);
        const fill = document.getElementById(`bar-fill-${i}`);
        const valText = document.getElementById(`bar-val-${i}`);
        
        const probPct = (probs[i] * 100).toFixed(1);
        fill.style.width = `${probPct}%`;
        valText.innerText = `${probPct}%`;
        
        if (i === maxValIdx && probs[i] > 0.05) {
            row.classList.add('active');
        } else {
            row.classList.remove('active');
        }
    }
}

// Softmax function for output layer
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// Perform forward propagation in JavaScript
function performInference() {
    if (!hasDrawn) return;

    // 1. Downsample drawing canvas (280x280) to preview canvas (28x28)
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
    previewCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

    // 2. Extract and preprocess pixels
    const imgData = previewCtx.getImageData(0, 0, 28, 28);
    const pixels = imgData.data;
    const input = new Float32Array(784);
    
    for (let i = 0; i < 784; i++) {
        // Red channel divided by 255 (inputs are normalized to [0, 1])
        input[i] = pixels[i * 4] / 255.0;
    }

    // 3. Forward Propagation through Feedforward MLP
    // Layer 1: Input (784) -> Hidden 1 (128)
    const h1 = new Float32Array(128);
    for (let j = 0; j < 128; j++) {
        let sum = model.b1[j];
        for (let i = 0; i < 784; i++) {
            sum += input[i] * model.W1[i][j];
        }
        h1[j] = Math.max(0, sum); // ReLU activation
    }

    // Layer 2: Hidden 1 (128) -> Hidden 2 (64)
    const h2 = new Float32Array(64);
    for (let j = 0; j < 64; j++) {
        let sum = model.b2[j];
        for (let i = 0; i < 128; i++) {
            sum += h1[i] * model.W2[i][j];
        }
        h2[j] = Math.max(0, sum); // ReLU activation
    }

    // Layer 3: Hidden 2 (64) -> Output (10)
    const out = new Array(10);
    for (let j = 0; j < 10; j++) {
        let sum = model.b3[j];
        for (let i = 0; i < 64; i++) {
            sum += h2[i] * model.W3[i][j];
        }
        out[j] = sum; // Linear activation before softmax
    }

    // Calculate probabilities via Softmax
    const probabilities = softmax(out);

    // Get predicted class (Argmax)
    let maxVal = -1;
    let predictedDigit = -1;
    for (let i = 0; i < 10; i++) {
        if (probabilities[i] > maxVal) {
            maxVal = probabilities[i];
            predictedDigit = i;
        }
    }

    // Update UI
    predictedDigitText.innerText = predictedDigit;
    updateConfidenceBars(probabilities);
}

// Debounce helper to avoid high CPU usage during continuous drawing
let timeoutId = null;
function debouncedPredict() {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
        performInference();
    }, 100);
}

// Initialize application on load
window.addEventListener('DOMContentLoaded', () => {
    initConfidenceBars();
    clearCanvas();
    loadModel();
});
