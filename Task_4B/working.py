import cv2
import numpy as np
import math
import csv

csv_filename = "xyz.csv"
lat_long_filename = "lat_long.csv"

detected_markers = []  # List to store detected markers
last_lat, last_lon = None, None

def calculate_distance(point1, point2):
    return int(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def detect_ArUco_details(image):
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
    shortest_distance = float('inf')
    closest_marker_id = None

    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0, 0, 255), -1)

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
                cv2.line(image, center, closest_center, (0, 255, 255), 2)
                length_text = f"Distance: {shortest_distance} pixels"
                cv2.putText(image, length_text, (int((center[0] + closest_center[0]) / 2), int((center[1] + closest_center[1]) / 2)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

                # Check if the closest marker has changed
                with open(csv_filename, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)

                    if closest_marker_id not in detected_markers:
                        detected_markers.append(closest_marker_id)

                        # Load lat_long.csv for comparison
                        with open(lat_long_filename, mode='r') as lat_long_file:
                            lat_long_reader = csv.reader(lat_long_file)
                            lat_long_dict = {str(row[0]): [row[1], row[2]] for row in lat_long_reader}

                        # Update xyz.csv with lat and lon values
                        if str(closest_marker_id) in lat_long_dict:
                            lat, lon = lat_long_dict[str(closest_marker_id)]

                        
                            # Append xyz.csv with lat and lon values
                            with open(csv_filename, mode='a', newline='') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow([lat, lon])

                            print(f"Closest ArUco ID: {closest_marker_id}, Lat: {lat}, Lon: {lon}")
                        else:
                            print(f"ArUco ID {closest_marker_id} not found in lat_long.csv")

    for ids, corner in ArUco_corners.items():
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)

        # cv2.line(image, center, (tl_tr_center_x, tl_tr_center_y), (255, 0, 0), 5)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0]) ** 2 + (tl_tr_center_y - center[1]) ** 2))
        cv2.putText(image, str(ids), (center[0] + int(display_offset / 2), center[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        angle = details[1]
        cv2.putText(image, str(angle), (center[0] - display_offset, center[1] - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

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
