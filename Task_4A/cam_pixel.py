import cv2

def event_identification(arena):
    # Convert the arena image to grayscale
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for i, contour in enumerate(contours):
        # Filter contours based on area
        if 1000 < cv2.contourArea(contour) < 4000:
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the event region from the original arena image
            event = arena[y:y + h, x:x + w]

            # Draw the contour area information on the image
            cv2.drawContours(arena, [contour], 0, (0, 255, 0), 2)
            cv2.putText(arena, f"Contour {i + 1}: {cv2.contourArea(contour)} pixels", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append the extracted event to the list
            event_list.append(event)

    # Swap the 3rd and 4th images in the list (0-based indexing)
    if len(event_list) >= 4:
        event_list[2], event_list[3] = event_list[3], event_list[2]

    return event_list

if __name__ == "__main__":
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change it if you have multiple cameras

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
            
            # Print the pixel count for each identified event
            print(f"Event {i + 1}: {event_image.size} pixels")

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
