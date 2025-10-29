import zipfile
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# === Step 1: Unzip the dataset ===
zip_path = '/content/archive.zip'
extract_dir = '/content/fer_dataset/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extracted folders:", os.listdir(extract_dir))

# === Step 2: Define dataset paths ===
train_dir = os.path.join(extract_dir, 'train')
test_dir = os.path.join(extract_dir, 'test')

print("Train folders:", os.listdir(train_dir))
print("Test folders:", os.listdir(test_dir))

# === Step 3: Load and preprocess images ===
emotion_labels = ['fear', 'sad', 'surprise', 'happy', 'neutral', 'angry', 'disgust']

def load_images_from_directory(directory, label_map, image_size=(48, 48)):
    X, y = [], []
    for label, emotion in enumerate(label_map):
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Missing folder: {emotion_dir}")
            continue
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            X.append(img)
            y.append(label)
    return np.array(X, dtype='float32'), np.array(y, dtype='int32')

X_train, y_train = load_images_from_directory(train_dir, emotion_labels)
X_test, y_test = load_images_from_directory(test_dir, emotion_labels)

# Normalize and reshape
X_train /= 255.0
X_test /= 255.0
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(emotion_labels))
y_test = to_categorical(y_test, num_classes=len(emotion_labels))

print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")

# === Step 4: Data Augmentation ===
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# === Step 5: CNN Model ===
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# === Step 6: Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# === Step 7: Train ===
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=25,
    steps_per_epoch=len(X_train) // 64,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# === Step 8: Plot Accuracy and Loss ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Step 9: Evaluate ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# === Step 10: Save the Model ===
model.save('emotion_recognition_model.h5')
print("Model saved successfully!")

# === Step 11: Prediction on Single Image ===
def predict_emotion(image_path):
    model = load_model('emotion_recognition_model.h5')
    emotion_labels = ['fear', 'sad', 'surprise', 'happy', 'neutral', 'angry', 'disgust']

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    predictions = model.predict(img)
    predicted_label = emotion_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    return predicted_label, confidence

# Example usage
image_path = "/content/13.jpeg"
pred_label, conf = predict_emotion(image_path)
print(f"Predicted Emotion: {pred_label}, Confidence: {conf:.2f}%")
