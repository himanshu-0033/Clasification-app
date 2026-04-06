const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('snapshot-canvas');
const captureBtn = document.getElementById('capture-btn');
const resultBox = document.getElementById('result-box');
const predictedClassEl = document.getElementById('predicted-class');
const predictedConfidenceEl = document.getElementById('predicted-confidence');
const cameraLens = document.querySelector('.camera-lens');

// 1. Initialize High-Definition Camera Stream
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user" // Forces front-facing camera on mobile default
            } 
        });
        videoElement.srcObject = stream;
    } catch (err) {
        console.error("Camera Hardware Request Error:", err);
        alert("Camera access denied! Please enable permissions to utilize the AI vision system.");
    }
}

// Boot up stream
initCamera();

// 2. Orchestrate capturing, ML inference request, and result display
captureBtn.addEventListener('click', async () => {
    // Modify UI state to 'Processing'
    const btnText = captureBtn.querySelector('.btn-text');
    const btnLoader = captureBtn.querySelector('.btn-loader');
    
    captureBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'block';
    
    // Enable visual scanning animation overlay
    cameraLens.classList.add('scanning');
    
    // Clear out any previous classification results
    resultBox.classList.add('hidden');

    try {
        // Draw the current live video frame perfectly into the invisible backing canvas
        const ctx = canvasElement.getContext('2d');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        // Reverse horizontally to undo the CSS mirror flip so the AI gets the photo un-mirrored
        ctx.translate(canvasElement.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        // Compress image into base64 format for network transit (0.8 quality jpeg)
        const imageBase64 = canvasElement.toDataURL('image/jpeg', 0.8);

        // Post request to backend Flask inference API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageBase64 })
        });
        
        const data = await response.json();

        if (response.ok) {
            // Processing Successful: Update GUI details dynamically
            predictedClassEl.textContent = data.prediction;
            predictedConfidenceEl.textContent = data.confidence;
            
            // Pop the result container into view
            resultBox.classList.remove('hidden');
        } else {
            console.error("Inference Error:", data.error);
            alert("AI Inference Error: " + (data.error || "System failed to resolve the image structure"));
        }

    } catch (error) {
        console.error("Network Topology Error:", error);
        alert("Failed to communicate with ML Backend Server. Please ensure app.py is actively running.");
    } finally {
        // Re-enable button state
        captureBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
        
        // Remove hologram scan animation
        cameraLens.classList.remove('scanning');
    }
});
