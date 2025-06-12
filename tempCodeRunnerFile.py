import tensorflow as tf

model = tf.keras.models.load_model('models/model1.h5')
print("Model input shape:", model.input.shape)