import cv2
import numpy as np
from matplotlib import pyplot as plt

def super_resolve(image):
    # Create the Super Resolution object
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # Load the pre-trained LapSRN model with a scale factor of 8
    model_path = "LapSRN_x8.pb"
    sr.readModel(model_path)
    sr.setModel("lapsrn", 8)

    # Upscale the image
    img_upscaled = sr.upsample(image)

    return img_upscaled

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

if __name__ == "__main__":
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

    # Display each identified event image in a separate window
    for i, event_image in enumerate(identified_events):
        # Upscale the event image
        event_image_upscaled = super_resolve(event_image)

        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Identified Event {i + 1}")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(event_image_upscaled, cv2.COLOR_BGR2RGB))
        plt.title(f"Upscaled Event {i + 1}")

        plt.show()

    # Display the arena image with contour area information
    plt.figure(figsize=(8, 4))
    plt.imshow(cv2.cvtColor(arena_image, cv2.COLOR_BGR2RGB))
    plt.title("Arena Image with Contour Area Information")
    plt.show()
