import cv2
import numpy as np
import math

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

def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0, 0, 255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)

        cv2.line(image, center, (tl_tr_center_x, tl_tr_center_y), (255, 0, 0), 5)
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

        # Scale down the frame by a factor of 0.8 on both x and y axes
        # frame = cv2.resize(frame, None, fx=0.8, fy=0.8)

        # Detect ArUco markers in the frame
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(frame)

        # Display the marked frame
        frame_marked = mark_ArUco_image(frame, ArUco_details_dict, ArUco_corners)
        cv2.imshow("Marked Frame", frame_marked)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
