import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def event_identification(arena):
    # Convert the arena image to grayscale
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for i, contour in enumerate(contours):
        # Filter contours based on area
        if 8000 < cv2.contourArea(contour) < 12000:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the event region from the original arena image
            event = arena[y:y + h, x:x + w]

            # Draw the contour area information on the image
            cv2.drawContours(arena, [contour], 0, (0, 255, 0), 2)
            cv2.putText(arena, f"Contour {i + 1}: {cv2.contourArea(contour)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append the extracted event to the list
            event_list.append(event)

    # Swap the 3rd and 4th images in the list (0-based indexing)
    if len(event_list) >= 4:
        event_list[2], event_list[3] = event_list[3], event_list[2]

    return event_list

def classify_event(image):
    ''' 
    Purpose:
    ---
    This function will load your trained model and classify the event from an image which is 
    sent as an input.

    Input Arguments:
    ---
    `image`: Image path sent by input file

    Returns:
    ---
    `event` : [ String ]
                          Detected event is returned in the form of a string

    Example call:
    ---
    event = classify_event(image_path)
    '''
    # Preprocess and resize the image to match the input size of your model
    image = cv2.resize(image, (224, 224))  # Resize the image to 224x224

    # Preprocess the image (you might need to adjust this based on your model's training preprocessing)
    image = tf.keras.applications.densenet.preprocess_input(image)  # Apply the same preprocessing as during training

    # Display the preprocessed image for debugging
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze())
    plt.title("Preprocessed Image")
    plt.show()

    # Make predictions using the loaded model
    predicted_probabilities = model.predict(image[np.newaxis, ...])

    # Find the predicted class index with the highest probability
    predicted_class = np.argmax(predicted_probabilities)

    # Get the detected event based on the class index
    event = event_names[predicted_class]
    return event


if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model("alien_attack_model.h5")
    event_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]

    # Load the arena image
    arena_image = cv2.imread("arena.png")

    # Convert the arena image to RGB (OpenCV loads images in BGR format)
    arena_image_rgb = cv2.cvtColor(arena_image, cv2.COLOR_BGR2RGB)

    # Call the event_identification function
    identified_events = event_identification(arena_image)

    # Display the original arena image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(arena_image_rgb)
    plt.title("Original Arena Image")

    # Display each identified event image and classify it
    for i, event_image in enumerate(identified_events):
        # Classify the event image
        event_class = classify_event(event_image)

        # Display the classified event image
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Identified Event {i + 1}: {event_class}")
        plt.show()

    # Display the arena image with contour area information
    plt.figure(figsize=(8, 4))
    plt.imshow(cv2.cvtColor(arena_image, cv2.COLOR_BGR2RGB))
    plt.title("Arena Image with Contour Area Information")
    plt.show()
