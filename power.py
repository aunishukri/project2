import cv2
import tensorflow as tf
import numpy as np

# Load model TensorFlow yang telah dilatih
model = tf.keras.models.load_model('model_anda.h5')  # Gantikan dengan model yang anda gunakan

# Nama kelas (label)
classes = ['Malaysia', 'UK', 'Japan', 'China']

# Mulakan webcam
cap = cv2.VideoCapture(0)

while True:
    # Ambil frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Tunjukkan video dalam window
    cv2.imshow("Webcam", frame)

    # Tunggu pengguna tekan spacebar untuk tangkap gambar
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Tekan space untuk tangkap gambar
        # Tukar imej kepada format yang sesuai untuk model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Ukuran bergantung kepada model anda
        img = np.expand_dims(img, axis=0)  # Menambah batch dimension
        img = img / 255.0  # Normalisasi imej

        # Klasifikasi imej
        predictions = model.predict(img)
        class_idx = np.argmax(predictions)  # Dapatkan index kelas yang paling tinggi
        class_name = classes[class_idx]  # Dapatkan nama negara berdasarkan index
        
        # Paparkan keputusan pada terminal dan skrin
        print(f"Detected: {class_name}")
        cv2.putText(frame, f"Detected: {class_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Tunjukkan imej dengan label
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)  # Tunggu sehingga pengguna menekan key untuk tutup

    # Jika tekan 'q' keluar dari webcam
    if key == ord('q'):
        break

# Lepaskan webcam dan tutup semua window
cap.release()
cv2.destroyAllWindows()
