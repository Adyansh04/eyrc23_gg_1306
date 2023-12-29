import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("alien_attack_model.h5")
event_names = {
    0: "combat",
    1: "destroyed_buildings",
    2: "fire",
    3: "human_aid_rehabilitation",
    4: "military_vehicles"
}

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_color_balance(image, alpha=1.2, beta=10):
    return cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

def preprocess_image(image):
    image = adjust_color_balance(image, alpha=1.2, beta=10)
    image = gamma_correction(image, gamma=1.2)
    image = cv2.medianBlur(image, 3)
    return image

def event_identification(arena):
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    event_list = []

    for i, contour in enumerate(contours):
        if 8000 < cv2.contourArea(contour) < 11000:
            x, y, w, h = cv2.boundingRect(contour)
            event = arena[y:y + h, x:x + w]
            event_list.append((x, y, w, h, event))

    # if len(event_list) >= 4:
    #     # Swap the positions of events C and D
    #     event_list[2], event_list[3] = event_list[3], event_list[2]

    return event_list

def classify_event(image, resolution):
    image = preprocess_image(image)
    image = cv2.resize(image, (224, 224))

    predicted_probabilities = model.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(predicted_probabilities)
    event = event_names[predicted_class]
    
    # Return the identified event as a dictionary
    return {"A": event}

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    set_camera_resolution(cap, 1920, 1080)

    while True:
        ret, frame = cap.read()
        identified_events = event_identification(frame.copy())

        identified_labels = {}  # Initialize an empty dictionary

        for i, (x, y, w, h, event_image) in enumerate(identified_events):
            resolution = (w, h)
            identified_labels[chr(ord("A") + i)] = classify_event(event_image, resolution)["A"]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the classified text inside the bounding box
            classified_text = identified_labels[chr(ord('A') + i)]
            cv2.putText(frame, f"{classified_text}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the identified labels in the terminal
        print("identified_labels = ",identified_labels)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
