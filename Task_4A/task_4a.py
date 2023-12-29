'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ 1306 ]
# Author List:		[ Adyansh Gupta, Chetas Hedaoo, Megha Datta, Harsh Mehta ]
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
                               
import cv2
import tensorflow as tf
import numpy as np

################# ADD UTILITY FUNCTIONS HERE #################

def set_camera_resolution(cap, width, height):
    """
    Purpose:
    ---
    Set the resolution of the camera capture
    
    Arguments:
    ---
    `cap` : cv2.VideoCapture
        object representing the camera capture
    `width` : int
        desired width of the frame
    `height` : int
        desired height of the frame

    Returns:
    ---
    None
    """

    # Set the width and height of the captured frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def gamma_correction(image, gamma=1.0):
    """
    Purpose:
    ---
    Apply gamma correction to an image
    
    Arguments:
    ---
    `image` : np.ndarray
        input image
    `gamma` : float
        gamma correction parameter

    Returns:
    ---
    `gamma_corrected_image` : np.ndarray
        gamma-corrected image
    """

    inv_gamma = 1.0 / gamma      # Calculate the inverse of gamma

    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")         # Create a lookup table for gamma correction


    return cv2.LUT(image, table)

def adjust_color_balance(image, alpha=1.2, beta=10):
    """
    Purpose:
    ---
    Adjust color balance using alpha and beta parameters

    
    Arguments:
    ---
    `image` : np.ndarray
        input image
    `alpha` : float
        alpha parameter for color adjustment
    `beta` : int
        beta parameter for color adjustment

    Returns:
    ---
    `color_adjusted_image` : np.ndarray
        color-adjusted image
    """

    return cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

def preprocess_image(image):
    """
    Purpose:
    ---
    Apply preprocessing steps to an image
    
    Arguments:
    ---
    `image` : np.ndarray
        input image

    Returns:
    ---
    `preprocessed_image` : np.ndarray
        preprocessed image
    """

    image = adjust_color_balance(image, alpha=1.2, beta=10)
    image = gamma_correction(image, gamma=1.2)
    image = cv2.medianBlur(image, 3)
    return image

def event_identification(arena):

    """
    Purpose:
    ---
    Identify events in an arena image
    
    Arguments:
    ---
    `arena` : np.ndarray
        input arena image

    Returns:
    ---
    `event_list` : list
        list of identified events in the arena
    """

    # Convert the arena image to grayscale
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)         

    # Apply thresholding to create a binary image    
    _, binary_image = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)     

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

    event_list = []          # List to store the identified events


    # Iterate through the contours and identify the events
    for i, contour in enumerate(contours):

        # Filter contours based on area
        if 8000 < cv2.contourArea(contour) < 11000:
            x, y, w, h = cv2.boundingRect(contour)       # Get the bounding box of the contour

            event = arena[y:y + h, x:x + w]     # Crop the event image from the arena image

            event_list.append((x, y, w, h, event))      # Append the event to the list

    if len(event_list) >= 4:
        # Swap the positions of events C and D
        event_list[2], event_list[3] = event_list[3], event_list[2]

    return event_list

def classify_event(image, resolution):
    """
    Purpose:
    ---
    Classify an event image and return the identified event
    
    Arguments:
    ---
    `image` : np.ndarray
        input event image
    `resolution` : tuple
        resolution of the event image

    Returns:
    ---
    `event_dict` : dict
        dictionary containing the identified event
    """
    # Preprocess the image
    image = preprocess_image(image)

    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))


    predicted_probabilities = model.predict(image[np.newaxis, ...])         # Predict the probabilities of each class

    predicted_class = np.argmax(predicted_probabilities)        # Get the class with the highest probability

    event = event_names[predicted_class]        # Get the name of the event

    # Return the identified event as a dictionary
    return {"A": event}
##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable
    
    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can 
    return the dictionary from a user-defined function and just call the 
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """  
    identified_labels = {}  
    
    cap = cv2.VideoCapture(0)           # Initialize the camera capture object

    set_camera_resolution(cap, 1920, 1080)


    while True:
        ret, frame = cap.read()

        # Crop the frame from 200 pixels from the left to 1500 pixels to the right
        frame = frame[:, 200:1500, :]

        identified_events = event_identification(frame.copy())

        for i, (x, y, w, h, event_image) in enumerate(identified_events):

            resolution = (w, h)
            identified_labels[chr(ord("A") + i)] = classify_event(event_image, resolution)["A"]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the classified text inside the bounding box
            classified_text = identified_labels[chr(ord('A') + i)]
            cv2.putText(frame, f"{classified_text}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the identified labels in the terminal
        scaled_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        print("identified_labels =", identified_labels)

        # Scale down the window to fit the cropped frame in the screen
        cv2.imshow("Camera Feed", scaled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":

    # Load the pre-trained model and define event names
    model = tf.keras.models.load_model("alien_attack_model.h5")
    event_names = {
        0: "combat",
        1: "destroyed_buildings",
        2: "fire",
        3: "human_aid_rehabilitation",
        4: "military_vehicles"
    }
    identified_labels = task_4a_return()
    print(identified_labels)