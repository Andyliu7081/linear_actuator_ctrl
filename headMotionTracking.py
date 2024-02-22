import cv2
import dlib
import numpy as np

def track_head_movement(model_path):
    """
    Tracks the head movement by detecting facial landmarks and allows setting initial and end positions manually.
    
    Parameters:
    - model_path: Path to the dlib facial landmark detection model file.
    
    Returns:
    - distance_moved: The distance moved by the head in pixels. Returns None if the start or end position was not set.
    """
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    cap = cv2.VideoCapture(0)

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
            nose_tip = (landmarks.part(33).x, landmarks.part(33).y)
            current_position = nose_tip
            cv2.circle(frame, nose_tip, 4, (0, 255, 0), -1)  # Visualize the nose tip

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            initial_position = current_position
            print("Start position set.")
        elif key == ord('e'):
            end_position = current_position
            print("End position set.")
            break
        elif key == ord('q'):
            print("Quitting without setting end position.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if initial_position and end_position:
        distance_moved = calculate_distance(initial_position, end_position)
        # print(f"Distance moved: {distance_moved} pixels")
        return distance_moved
    else:
        print("Start and/or end position not set.")
        return None

