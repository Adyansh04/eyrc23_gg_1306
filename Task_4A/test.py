####################### IMPORT MODULES #######################
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

##############################################################
event_names = ["fire", "destroyed_buildings", "human_aid_rehabilitation", "military_vehicles", "combat"]

################# ADD UTILITY FUNCTIONS HERE #################

def event_identification(arena):
    ''' 
    Purpose:
    ---
    This function will identify events on the printed arena image and extract them as separate images.

    Input Arguments:
    ---
    `arena`: Image of the arena detected by arena_image()

    Returns:
    ---
    `event_list`: List containing the extracted event images.

    Example call:
    ---
    event_list = event_identification(arena)
    '''
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for contour in contours:
        if 1000 < cv2.contourArea(contour) < 4400:
            x, y, w, h = cv2.boundingRect(contour)
            event = arena[y:y + h, x:x + w]
            event_list.append(event)

    return event_list


def classify_event(image, model):
    ''' 
    Purpose:
    ---
    This function will load the trained model and classify the event from an image.

    Input Arguments:
    ---
    `image`: Event image for classification.
    `model`: Trained model for event classification.

    Returns:
    ---
    `event`: Detected event as a string.

    Example call:
    ---
    event = classify_event(event_image, model)
    '''
    # Preprocess and resize the image to match the input size of the model
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)

    # Make predictions using the loaded model
    predicted_probabilities = model.predict(image[np.newaxis, ...])

    # Find the predicted class index with the highest probability
    predicted_class = np.argmax(predicted_probabilities)

    # Get the detected event based on the class index
    event = event_names[predicted_class]

    return event


def draw_bounding_box(image, label, color=(0, 255, 0)):
    '''
    Purpose:
    ---
    This function draws a bounding box and displays the classified label on the image.

    Input Arguments:
    ---
    `image`: Input image.
    `label`: Classified label.
    `color`: Bounding box and label text color.

    Returns:
    ---
    None.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Get the size of the text box
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size[0], text_size[1]

    # Draw bounding box
    cv2.rectangle(image, (0, 0), (text_width + 10, text_height + 10), color, -1)
    cv2.putText(image, label, (5, text_height + 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


##############################################################

def task_4a_return():
    identified_labels = {}

    model = load_model("alien_attack_model.h5")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        event_list = event_identification(frame)

        for index, event_image in enumerate(event_list):
            detected_event = classify_event(event_image, model)
            identified_labels[chr(65 + index)] = detected_event
            draw_bounding_box(frame, detected_event)

        cv2.imshow("Event Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return identified_labels

if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)