import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Fingertip IDs for counting (adjust based on your hand landmark structure)
finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp.hands.HandLandmark.PINKY_TIP]

# Function to count raised fingers
def count_fingers(landmarks):
    # Count the number of raised fingertips
    raised_fingers = 0
    for tip_id in finger_tips:
        if landmarks[tip_id].y < landmarks[mp_hands.HandLandmark.WRIST].y:
            raised_fingers += 1
    return raised_fingers

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB for MediaPipe hands
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Make image writable for processing

    # Detect hands
    results = mp_hands.process(image)

    # Draw hand landmarks and count fingers
    image.flags.writeable = True  # Restore writability
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers and display the number
            finger_count = count_fingers(hand_landmarks.landmark)
            cv2.putText(image, str(finger_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)  # Display finger count in green

    cv2.imshow('Finger Counter', image)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
