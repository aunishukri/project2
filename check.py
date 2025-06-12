import tensorflow as tf

for i in range(1, 5):
    model = tf.keras.models.load_model(f'models/model{i}.h5')
    print(f"Model{i} Input Shape: {model.input_shape}")
