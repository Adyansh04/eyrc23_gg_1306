arena_path = "sample.png"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

def event_identification(arena_path, show_images=True):
    '''
    Purpose:
    ---
    This function will select the events on an arena image and extract them as
    separate images.

    Input Arguments:
    ---
    `arena_path`: Path to the arena image.
    `show_images`: Flag to control whether to display the extracted event images.

    Returns:
    ---
    `event_list`: List of extracted event images as NumPy arrays.

    Example call:
    ---
    event_list = event_identification(arena_path)
    '''
    # Read the arena image
    arena = cv2.imread(arena_path)

    # Convert the image to grayscale for contour detection
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    # Use thresholding to detect the white border
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []  # List to store extracted event images

    for contour in contours:
        # Filter out small contours (noise)
        if 1000 < cv2.contourArea(contour) < 4400:
            # Get the coordinates of the bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the event region from the original arena image
            event = arena[y:y + h, x:x + w]

            # Append the extracted event to the list
            event_list.append(event)

    # Visualize the extracted event images
    if show_images:
        for i, event in enumerate(event_list):
            plt.figure()
            plt.imshow(cv2.cvtColor(event, cv2.COLOR_BGR2RGB))
            plt.title(f'Event {i+1}')
            plt.show()

    return event_list


event_identification(arena_path)
