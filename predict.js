async function captureAndSend() {
    for (let i = 1; i <= 4; i++) {
        const video = document.getElementById(`webcam${i}`);
        const resultElement = document.getElementById(`result${i}`);
        
        // 1. Early validation
        if (!video) {
            console.warn(`Video element webcam${i} not found`);
            if (resultElement) resultElement.textContent = `Webcam ${i} not found`;
            continue;
        }

        // 2. Check video readiness more reliably
        if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
            console.warn(`Webcam ${i} not ready`);
            if (resultElement) resultElement.textContent = `Webcam ${i} initializing...`;
            continue;
        }

        // 3. Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;  // Use original resolution
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 4. Create two versions:
        //    - High-res for OCR (original size)
        //    - Low-res (224x224) for classification
        const hiResCanvas = document.createElement('canvas');
        hiResCanvas.width = canvas.width;
        hiResCanvas.height = canvas.height;
        hiResCanvas.getContext('2d').drawImage(video, 0, 0);
        
        const loResCanvas = document.createElement('canvas');
        loResCanvas.width = 224;
        loResCanvas.height = 224;
        loResCanvas.getContext('2d').drawImage(video, 0, 0, 224, 224);

        // 5. Prepare data
        const hiResImage = hiResCanvas.toDataURL("image/jpeg", 0.9); // Higher quality for OCR
        const loResImage = loResCanvas.toDataURL("image/jpeg", 0.8); // Good quality for classification

        if (!hiResImage || hiResImage.length < 1000) {
            if (resultElement) resultElement.textContent = 'Camera feed empty';
            continue;
        }

        const modelName = `model${i}`;

        // 6. Show loading state
            if (resultElement) {
                resultElement.innerHTML = `<div style="color: blue;">🔄 Processing ${modelName}...</div>`;
            }
        
        try {
            // 6. Send both images to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    hi_res_image: hiResImage,  // For OCR
                    lo_res_image: loResImage,  // For classification
                    model_name: modelName
                })
            });

             if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            // 7. Handle response

            const result = await response.json();

            // 7. Enhanced result display
            if (result.error) {
                if (resultElement) {
                    resultElement.innerHTML = `Error: ${result.error}<br>`;
                    if (result.text) resultElement.innerHTML += `Text: ${result.text}`;
                }
            } else {
                if (resultElement) {
                    let statusColor = result.status === 'OK' ? 'green' : (result.status === 'NG' ? 'red' : 'black');
                    resultElement.innerHTML = `
                        <strong>${modelName}:</strong> ${result.prediction}<br>
                        ${result.text ? `<strong>Text:</strong> ${result.text}` : ''}
                        ${result.status ? `<strong>Status:</strong> <span style="color:${statusColor}">${result.status}</span>` : ''}
                    `;
                }
            }
        } catch (err) {
            console.error(`Prediction failed for ${modelName}:`, err);
            if (resultElement) {
                resultElement.innerHTML = `Prediction failed<br>${err.message}`;
            }
        }
    }
}

// 8. Add retry mechanism with delay
let retryCount = 0;
const MAX_RETRIES = 3;
let isProcessing = false;

async function captureWithRetry() {
    try {
        await captureAndSend();
        retryCount = 0;
    } catch (err) {
        if (retryCount < MAX_RETRIES) {
            retryCount++;
            setTimeout(captureWithRetry, 1000 * retryCount);
        }
    }
}

// 9. Call this instead of direct captureAndSend()
captureWithRetry();