import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

def event_identification(arena_path, show_images=True):
    arena = cv2.imread(arena_path)
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for contour in contours:
        if 1000 < cv2.contourArea(contour) < 4400:
            x, y, w, h = cv2.boundingRect(contour)
            event = arena[y:y + h, x:x + w]
            event_list.append((x, y, w, h, event))

    if show_images:
        for i, (_, _, _, _, event_image) in enumerate(event_list):
            # Add label indicating the order (A, B, C, D, E)
            plt.figure()
            plt.imshow(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Event {chr(65 + i)}')
            plt.show()

    return event_list

def classify_event(image, model):
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    predicted_probabilities = model.predict(image[np.newaxis, ...])

    event_names = ["fire", "destroyed_buildings", "human_aid_rehabilitation", "military_vehicles", "combat"]
    predicted_class = np.argmax(predicted_probabilities)
    event = event_names[predicted_class]

    return event

def draw_bounding_box(image, coordinates, label):
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color (R=0, B=0, G=255)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green color (R=0, B=0, G=255)
    return image

def task_4a_return():
    model = load_model("alien_attack_model.h5")
    event_names = ["fire", "destroyed_buildings", "human_aid_rehabilitation", "military_vehicles", "combat"]
    detected_labels = []
    identified_labels = {}

    # Load the sample image
    sample_image_path = "sample.png"
    sample_image = cv2.imread(sample_image_path)

    # Identify events on the sample image
    event_list = event_identification(sample_image_path, show_images=False)

    # Iterate through each event, classify, and draw bounding box with label
    for index, (x, y, w, h, event_image) in enumerate(event_list):
        detected_event = classify_event(event_image, model)
        detected_labels.append(detected_event)
        identified_labels[chr(65 + index)] = detected_event

        # Draw bounding box and label on the sample image
        sample_image = draw_bounding_box(sample_image, (x, y, w, h), detected_event)

    # Add label indicating the order of images (A, B, C, D, E)
    cv2.putText(sample_image, "ABCDE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result image with bounding boxes, labels, and order
    cv2.imshow('Classified Events', sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return identified_labels

if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)
