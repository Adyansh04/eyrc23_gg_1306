import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Path to the input image
input_image_path = r"C:\Users\gupta\Desktop\Eyantra_GG(2023)\geoguide_1306\Task_2B\training_new\Fire\fire.png"

# Directory to save augmented images
output_directory = r"C:\Users\gupta\Desktop\Eyantra_GG(2023)\geoguide_1306\Task_2B\training_new\Fire"

# Load the input image
img = load_img(input_image_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Create an instance of the ImageDataGenerator with desired augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images and save to the output directory
num_augmented_images = 50
for _ in range(num_augmented_images):
    augmented_images = datagen.flow(x, batch_size=1, save_to_dir=output_directory, save_prefix='aug', save_format='jpg')
    augmented_image = augmented_images.next()[0]  # Extract the first image from the batch

# The loop above will generate and save 50 augmented images without displaying them.
