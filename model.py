import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define dataset directories
main_directory = 'fire_dataset'
training_directory = os.path.join(main_directory, 'Desktop/PyroSight/fire_dataset/Train')
validation_directory = os.path.join(main_directory, 'Desktop/PyroSight/fire_dataset/Validate')

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale = 1/255,                                                            # Normalize pixel values
    rotation_range = 30,                                                        # Random rotation
    width_shift_range = 0.2,                                                    # Horizontal shift
    height_shift_range = 0.2,                                                   # Vertical shift
    zoom_range = 0.2,                                                           # Zoom
    horizontal_flip = True                                                      # Random horizontal flip
)

# Validation data generator (no augmentation, just rescale)
validation_datagen = ImageDataGenerator(rescale = 1/255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    training_directory,
    target_size = (128, 128),                                                    # resize images to 128x128
    batch_size = 32,
    class_mode = 'binary'                                                        # Binary classification
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_directory,
    target_size = (128, 128),
    batch_size = 32,                                                             # CNNs like to take in bacthes, so we give them what they expect
    class_mode = 'binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape=(128, 128, 3)),         # 32 is the number of filters, 3x3 is the grid size, 128x128 image size, 3 is colors red, green, blue
    MaxPooling2D(pool_size = (2, 2)),                                           # 2x2 is the grid size
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')                                            # Sigmoid for binary classification
])

# compile the model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Add early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

# train the model
history = model.fit(
    train_generator,
    epochs = 20,
    validation_data = validation_generator,
    callbacks = [early_stopping]
)

# save the model
model.save('fire_detection_model.h5')

# Plot training history
plt.figure(figsize = (12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
