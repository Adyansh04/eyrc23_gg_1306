Team ID = 1306
Trained weights drive link = "https://drive.google.com/file/d/1JWtxzIkf4WcVuBIjhbaKE7ILugvEFD2F/view?usp=sharing" 
###############################################################################
'''
Please write the complete steps taken by your team explaining how you completed Task 2C. It is adviced to be as elaborate as possible.

Step 1: Training the Event Classification Model

In Task 2B, we trained a deep learning model to classify events. The details of this model training are not present in the script but are essential to the completion of Task 2C.

Step 2: Data Collection and Organization

In this step, we used an arena image to simulate the scenario. This image is loaded in the script using the arena_image(arena_path) function, which performs the following:

Reads the generated image specified by the arena_path variable.

Resizes the image to a fixed size of 700x700 pixels.

Step 3: Image Detection in the Arena

We used the OpenCV library (cv2) to perform image processing and contour detection on the arena image.

The arena image is converted to grayscale for contour detection. Thresholding is applied to detect the white border.

The script uses the event_identification(arena) function to identify events in the arena image. This function filters out small contours (noise) and extracts event regions as separate images. The extracted events are stored in the event_list.

Step 4: Event Classification

The classify_event(image) function is used to classify each event image. This function loads the pre-trained deep learning model.
The event images are processed and fed into the model for classification.
The function makes predictions using the loaded model and selects the detected event based on the class index with the highest probability.
The detected event is returned as a string.

Step 5: Output Processing

The classification(event_list) function processes the detected events. It iterates through the extracted event images and classifies them using the classify_event(image) function.
The detected events are appended to the detected_list and stored in a predefined list format.

Step 6: Running the Script

The script is executed using the main() function, which handles the entire event detection process.
The script takes care of input (team ID), image processing, event classification, and writing the output to a file.
