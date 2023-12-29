import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

# Define class names and labels
class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
class_labels = {class_name: i for i, class_name in enumerate(class_names)}

# Define custom preprocessing functions
def adjust_color_balance(image, alpha=1.2, beta=10):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return np.clip(result, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma=1.2):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def preprocess_image(image):
    # Apply your custom preprocessing steps
    image = adjust_color_balance(image, alpha=1.2, beta=10)
    image = gamma_correction(image, gamma=1.2)
    image = cv2.medianBlur(image, 3)  # Uncomment if you want to apply median blur
    return image

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

train_data_dir = "training"
test_data_dir = "testing"

train_data_gen = ImageDataGenerator(rescale=None,preprocessing_function=preprocess_image)
test_data_gen = ImageDataGenerator(rescale=None,preprocessing_function=preprocess_image)

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

# Fine-tune more layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

model = keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.6),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(class_names), activation='softmax')
])

# Use class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights_dict = {i: class_weights[i] for i in range(len(class_names))}
print("Class Weights:", class_weights_dict)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Training with a learning rate scheduler
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch):
    return 0.0001 * np.exp(-epoch / 10)

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Use class weights during training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[lr_callback, early_stopping],
    class_weight=class_weights_dict 
)

# Save the trained model
model.save("alien_attack_model.h5")

# Load the trained model for testing
model = keras.models.load_model("alien_attack_model.h5")

# Prepare the test data
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
