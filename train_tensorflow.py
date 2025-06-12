import scipy
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore

# Menyediakan dataset (contohnya, menggunakan ImageDataGenerator untuk dataset folder)
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True,
                                   validation_split=0.2)  # Gantikan dengan 'validation_split' jika dataset dibahagi

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\User\OneDrive\Desktop\project2\dataset\Webcam4',  # Gantikan dengan folder dataset latihan
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # Gantikan dengan 'validation' untuk set pengesahan
    shuffle=True  # Mengacak data untuk mengelakkan overfitting
)
val_generator = train_datagen.flow_from_directory(
    r'C:\Users\User\OneDrive\Desktop\project2\dataset\Webcam4',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model CNN dasar untuk klasifikasi imej
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Gantikan '3' dengan jumlah kelas anda
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(train_generator, epochs=10,validation_data=val_generator)

# Simpan model dalam format .h5
model.save('models/model4.h5')  # Gantikan dengan nama model yang sesuai

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Accuracy & Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('training_graph.png')
