import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to count fingers
def count_fingers(hand_landmarks):
    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    fingers_up = []

    tips = [4, 8, 12, 16, 20]  # Landmark indexes for fingertips
    pip_joints = [2, 6, 10, 14, 18]  # Joints before fingertips
    
    for i in range(5):
        if i == 0:  # Thumb
            if hand_landmarks.landmark[tips[i]].x < hand_landmarks.landmark[pip_joints[i]].x:
                fingers_up.append(fingers[i])
        else:  # Other fingers
            if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pip_joints[i]].y:
                fingers_up.append(fingers[i])
    
    return fingers_up

# Start capturing video
cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count raised fingers
                raised_fingers = count_fingers(hand_landmarks)
                text = f"Fingers Up: {', '.join(raised_fingers) if raised_fingers else 'None'}"
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                
                # Display detected fingers
                cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show output
        cv2.imshow("Finger Counter", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
