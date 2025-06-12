import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create directories
os.makedirs('models', exist_ok=True)

print("="*60)
print("HYBRID CNN + RANDOM FOREST TRAINING")
print("="*60)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load data
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\User\OneDrive\Desktop\project2\dataset\Webcam3',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    r'C:\Users\User\OneDrive\Desktop\project2\dataset\Webcam3',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Number of classes: {train_generator.num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Step 1: Create CNN Feature Extractor
print("\n" + "="*40)
print("STEP 1: TRAINING CNN FEATURE EXTRACTOR")
print("="*40)

feature_extractor = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),  # This will be our feature vector
])

# Complete CNN model for initial training
cnn_model = models.Sequential([
    feature_extractor,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train CNN briefly to learn good features
history = cnn_model.fit(
    train_generator,
    epochs=10,  # Fewer epochs for feature learning
    validation_data=val_generator,
    verbose=1
)

# Step 2: Extract Features using trained CNN
print("\n" + "="*40)
print("STEP 2: EXTRACTING FEATURES")
print("="*40)

def extract_features_and_labels(generator, feature_extractor):
    """Extract features using the trained CNN feature extractor"""
    features = []
    labels = []
    
    # Reset generator
    generator.reset()
    
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        # Extract features using the feature extractor
        batch_features = feature_extractor.predict(batch_x, verbose=0)
        features.extend(batch_features)
        # Convert one-hot labels to class indices
        labels.extend(np.argmax(batch_y, axis=1))
        
        if i % 10 == 0:
            print(f"Processed batch {i+1}/{len(generator)}")
    
    return np.array(features), np.array(labels)

# Extract features for training and validation
print("Extracting training features...")
train_features, train_labels = extract_features_and_labels(train_generator, feature_extractor)

print("Extracting validation features...")
val_features, val_labels = extract_features_and_labels(val_generator, feature_extractor)

print(f"Training features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")

# Combine all data for train_test_split
all_features = np.concatenate([train_features, val_features])
all_labels = np.concatenate([train_labels, val_labels])

# Step 3: Train RandomForest on extracted features
print("\n" + "="*40)
print("STEP 3: TRAINING RANDOM FOREST")
print("="*40)

# Split data
trainX, testX, trainY, testY = train_test_split(
    all_features, all_labels, 
    test_size=0.25, 
    random_state=42,
    stratify=all_labels
)

print(f"Training set size: {trainX.shape[0]}")
print(f"Test set size: {testX.shape[0]}")

# Train RandomForest
rf_model = RandomForestClassifier(
    n_estimators=200,  # Increased for better performance
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

print("Training Random Forest...")
rf_model.fit(trainX, trainY)

# Evaluate RandomForest
rf_accuracy = rf_model.score(testX, testY)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# Detailed evaluation
y_pred = rf_model.predict(testX)
class_names = list(train_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(testY, y_pred, target_names=class_names))

# Step 4: Save models
print("\n" + "="*40)
print("STEP 4: SAVING MODELS")
print("="*40)

# Save CNN feature extractor
feature_extractor.save('models/cnn_feature_extractor.h5')
print("CNN feature extractor saved as 'cnn_feature_extractor.h5'")

# Save complete CNN model
cnn_model.save('models/complete_cnn_model.h5')
print("Complete CNN model saved as 'complete_cnn_model.h5'")

# Save RandomForest model
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Random Forest model saved as 'random_forest_model.pkl'")

# Save class indices for later use
with open("models/class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print("Class indices saved as 'class_indices.pkl'")

# Save feature data for future use (optional)
np.save("models/extracted_features.npy", all_features)
np.save("models/extracted_labels.npy", all_labels)
print("Feature data saved as .npy files")

# Step 5: Visualization
print("\n" + "="*40)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*40)

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Feature Extractor Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Feature Extractor Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(testY, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('hybrid_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('Random Forest Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.grid(True)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"CNN Feature Extractor trained for {len(history.history['accuracy'])} epochs")
print(f"Final CNN Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Models saved in 'models/' directory")
print("="*60)