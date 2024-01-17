import cv2
import numpy as np
import math
import csv

csv_filename = "live_loc.csv"
lat_long_filename = "lat_long.csv"

# List to store detected markers
detected_markers = []

# Variable to store the ID of the last closest marker
last_closest_marker_id = None

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1: First point as [x, y]
    - point2: Second point as [x, y]

    Returns:
    - Distance between the two points (integer)
    """
    return int(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def detect_ArUco_details(image):
    """
    Detect ArUco markers in an image and retrieve their details.

    Parameters:
    - image: Input image (BGR format)

    Returns:
    - ArUco_details_dict: Dictionary with ArUco ID as key and details (center, angle) as value
    - ArUco_corners: Dictionary with ArUco ID as key and corner points as value
    """
    ArUco_details_dict = {}
    ArUco_corners = {}

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    # Detect ArUco markers in the image
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    # Check if markers are detected
    if markerIds is not None:
        # Loop through detected markers
        for i in range(len(markerIds)):
            marker_id = int(markerIds[i][0])
            corner_points = markerCorners[i][0].tolist()

            # Calculate the center coordinates
            center_x = int(np.mean(corner_points, axis=0)[0])
            center_y = int(np.mean(corner_points, axis=0)[1])

            # Calculate the angle from the vertical
            corner1 = corner_points[0]
            corner2 = corner_points[1]
            del_x = corner2[0] - corner1[0]
            del_y = corner2[1] - corner1[1]
            angle = math.degrees(math.atan2(del_y, del_x))

            # Add details to the dictionary
            ArUco_details_dict[marker_id] = [[center_x, center_y], int(angle)]
            ArUco_corners[marker_id] = corner_points

    return ArUco_details_dict, ArUco_corners

def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners, marker_id_to_track=100):
    """
    Mark ArUco markers on an image and update the live_loc.csv file.

    Parameters:
    - image: Input image (BGR format)
    - ArUco_details_dict: Dictionary with ArUco ID as key and details (center, angle) as value
    - ArUco_corners: Dictionary with ArUco ID as key and corner points as value
    - marker_id_to_track: ID of the marker to track

    Returns:
    - Marked image
    """
    global last_closest_marker_id  # Declare as global

    shortest_distance = float('inf')
    closest_marker_id = None

    for ids, details in ArUco_details_dict.items():
        center = details[0]

        # Track distance for the specified marker (id 100)
        if ids == marker_id_to_track:
            for other_ids, other_details in ArUco_details_dict.items():
                if other_ids != marker_id_to_track:
                    other_center = other_details[0]
                    distance = calculate_distance(center, other_center)

                    # Check for the shortest distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        closest_marker_id = other_ids

            # Draw a line between the specified marker and the closest marker
            if closest_marker_id is not None:
                closest_center = ArUco_details_dict[closest_marker_id][0]

                # Check if the closest marker has changed
                if closest_marker_id != last_closest_marker_id:
                    last_closest_marker_id = closest_marker_id

                    # Load lat_long.csv for comparison
                    with open(lat_long_filename, mode='r') as lat_long_file:
                        lat_long_reader = csv.reader(lat_long_file)
                        lat_long_dict = {str(row[0]): [row[1], row[2]] for row in lat_long_reader}

                    # Update live_loc.csv with lat and lon values
                    if str(closest_marker_id) in lat_long_dict:
                        lat, lon = lat_long_dict[str(closest_marker_id)]

                        # Write to live_loc.csv with lat and lon values
                        with open(csv_filename, mode='w', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(["lat", "lon"])  # Write header
                            csv_writer.writerow([lat, lon])

                        print(f"Closest ArUco ID: {closest_marker_id}, Lat: {lat}, Lon: {lon}")
                    else:
                        print(f"ArUco ID {closest_marker_id} not found in lat_long.csv")

    return image

if __name__ == "__main__":
    # Open the webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is captured successfully
        if not ret:
            print("Error capturing frame")
            break

        # Detect ArUco markers in the frame
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(frame)

        # Display the marked frame
        frame_marked = mark_ArUco_image(frame, ArUco_details_dict, ArUco_corners, marker_id_to_track=100)
        cv2.imshow("Marked Frame", frame_marked)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
