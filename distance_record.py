import cv2
import math
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Update the path as necessary

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

initial_position = None
end_position = None
current_position = None

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # For simplicity, use the tip of the nose (landmark 33) as the tracking point
        nose_tip = (landmarks.part(33).x, landmarks.part(33).y)
        current_position = nose_tip
        cv2.circle(frame, nose_tip, 4, (0, 255, 0), -1)  # Draw the nose tip for visualization

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Set initial_position on 's' key press
        if current_position:
            initial_position = current_position
            print("Start position set.")
    elif key == ord('e'):  # Set end_position on 'e' key press
        if current_position:
            end_position = current_position
            print("End position set.")
    elif key == ord('q'):  # Quit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()

if initial_position and end_position:
    distance_moved = calculate_distance(initial_position, end_position)
    print(f"Distance moved: {distance_moved} pixels")
else:
    print("Start and/or end position not set.")