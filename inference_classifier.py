import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture and Mediapipe Hand solution
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Thank You'
}

# Expected feature length
expected_length = 84

# String to store the detected letters
detected_letters = ""

# Timer variables for handling delays
last_detection_time = time.time()
letter_detected = False  # Flag to ensure only one letter is detected per click

# Flag to control when detection starts/stops
detection_started = False

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    predicted_character = None  # Initialize variable for predicted character

    # Process the hand detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmarks
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Adjust data_aux to match the expected length
        if len(data_aux) < expected_length:
            data_aux += [0] * (expected_length - len(data_aux))  # Pad with zeros
        elif len(data_aux) > expected_length:
            data_aux = data_aux[:expected_length]  # Truncate to expected length

        # Predict the character
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # If detection is started, and hand is detected, then we handle adding letters
        if detection_started and not letter_detected:
            detected_letters += predicted_character
            letter_detected = True  # Mark that a letter was detected
            print(f"Detected Letters: {detected_letters}")  # Print the detected letters in terminal

    else:
        # If no hand is detected, don't add any letter
        letter_detected = False  # Reset the letter detection flag

    # Display the predicted letter on the screen with color
    # Choose a color for the predicted letter (e.g., green color for the letter)
    color = (0, 255, 0)  # Green color for the predicted letter (BGR format)
    
    # Display the predicted letter in color
    cv2.putText(frame, f"Predicted Letter: {predicted_character if predicted_character else 'None'}", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Display the detected letters
    cv2.putText(frame, f"Detected Letters: {detected_letters}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display instructions for starting/stopping detection
    cv2.putText(frame, f"Press 's' to start/stop detection", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Handle key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):  # Start/stop detection when 's' is pressed
        detection_started = not detection_started  # Toggle the detection state
        if detection_started:
            print("Detection Started")
        else:
            print("Detection Stopped")

    if key == ord(' '):
         if detection_started:
            detected_letters += ' '
            print("Space added")

    if key == ord('`'):
         with open("detected_letters.txt", "w") as file:
            file.write(detected_letters)
         print("Detected letters saved to detected_letters.txt")

    
    

if key == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
