// webcam.js

let webcam1, webcam2, webcam3, webcam4;

// Setup webcam untuk setiap webcam
async function setupWebcam(id) {
    const videoElement = document.getElementById(id);
    const webcam = new tmImage.Webcam(200, 150, true);
    await webcam.setup();
    await webcam.play();
    videoElement.appendChild(webcam.canvas);

    return webcam;
}

// Fungsi untuk inisialisasi semua webcam
async function setupWebcams() {
    webcam1 = await setupWebcam("webcam1");
    webcam2 = await setupWebcam("webcam2");
    webcam3 = await setupWebcam("webcam3");
    webcam4 = await setupWebcam("webcam4");
}

// Panggil fungsi setupWebcams semasa page load
window.onload = setupWebcams;
