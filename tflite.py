import tensorflow as tf

model = tf.keras.models.load_model('model1.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan ke file
with open('model1.tflite', 'wb') as f:
    f.write(tflite_model)
