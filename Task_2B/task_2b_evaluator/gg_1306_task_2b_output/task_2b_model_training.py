import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define class names and labels
class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
class_labels = {class_name: i for i, class_name in enumerate(class_names)}

# Define image and batch size
image_size = (224, 224)  # Changed to match DenseNet201 input size
batch_size = 32

# Data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data_dir = "training"
test_data_dir = "testing" 

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=class_names
)

validation_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=class_names
)

# Build and compile the model (DenseNet201)
base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

model = keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),  
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
for layer in base_model.layers[-10:]:  
    layer.trainable = True

# Training with a learning rate scheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch):
    return 0.001 * np.exp(-epoch / 10)

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[lr_callback, early_stopping]
)

# Save the trained model
model.save("alien_attack_model.h5")

# Load the trained model for testing
model = keras.models.load_model("alien_attack_model.h5")

# Prepare the test data
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# Make predictions on test data
test_predictions = model.predict(test_generator)

# Convert predictions to class labels
predicted_labels = [class_names[i] for i in np.argmax(test_predictions, axis=1)]

# Display the predicted class labels for each test image
for i, image_path in enumerate(test_generator.filepaths):
    filename = os.path.basename(image_path)
    class_name = predicted_labels[i]
    print(f"Image: {filename}, Predicted Class: {class_name}")
