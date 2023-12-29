import cv2
import numpy as np

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_color_balance(image, alpha=1.2, beta=10):
    return cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

def event_identification(arena):
    # Convert the arena image to grayscale
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for i, contour in enumerate(contours):
        # Filter contours based on area
        if 6000 < cv2.contourArea(contour) < 15000:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the event region from the original arena image
            event = arena[y:y + h, x:x + w]

            # Uncomment the line below to use median blur for noise reduction
            event = cv2.medianBlur(event, 3)

            # Apply gamma correction
            event = gamma_correction(event, gamma=1.2)

            # Apply histogram equalization
            # event = histogram_equalization(event)

            # Apply CLAHE
            # event = clahe(event)

            # Adjust color balance
            event = adjust_color_balance(event, alpha=1.2, beta=10)

            # Draw the contour area information on the original image
            cv2.drawContours(arena, [contour], 0, (0, 255, 0), 2)
            cv2.putText(arena, f"Contour {i + 1}: {cv2.contourArea(contour)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append the processed event to the list
            event_list.append(event)

    # Swap the 3rd and 4th images in the list (0-based indexing)
    if len(event_list) >= 4:
        event_list[2], event_list[3] = event_list[3], event_list[2]

    return event_list

if __name__ == "__main__":
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change it if you have multiple cameras

    # Set the desired resolution (1080p)
    set_camera_resolution(cap, 1920, 1080)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Call the event_identification function
        identified_events = event_identification(frame)

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Display each identified event image with bounding box in a separate window
        for i, event_image in enumerate(identified_events):
            # Display the event image with bounding box
            cv2.imshow(f"Identified Event {i + 1}", event_image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
