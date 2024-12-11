import cv2
import mediapipe as mp

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Set up Video Capture
cap = cv2.VideoCapture(0)  # Use webcam (0 for default)

# Function to detect specific gestures
def detect_gesture(landmarks):
    # Extract landmark positions for gesture detection
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    gesture = ""

    # Thumbs Up (Thumb up, others curled)
    if thumb_tip.y < thumb_base.y and \
       index_tip.y > index_mcp.y and \
       middle_tip.y > middle_mcp.y and \
       ring_tip.y > ring_mcp.y and \
       pinky_tip.y > pinky_mcp.y:
        gesture = "Thumbs Up"

    # Victory (Peace sign, index and middle up, others curled)
    elif index_tip.y < index_mcp.y and \
         middle_tip.y < middle_mcp.y and \
         ring_tip.y > ring_mcp.y and \
         pinky_tip.y > pinky_mcp.y:
        gesture = "Victory"

    # Palm (All fingers extended)
    elif index_tip.y < index_mcp.y and \
         middle_tip.y < middle_mcp.y and \
         ring_tip.y < ring_mcp.y and \
         pinky_tip.y < pinky_mcp.y:
        gesture = "Palm"

    return gesture

# Set the font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Process the Image for Hand Detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get results
    results = hands.process(rgb_frame)

    # Initialize hand count
    hand_count = 0

    # Check for detected hands
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)

        # To prevent overlap, start positioning gesture labels based on the hand index
        gesture_offset_y = 40  # Distance between each gesture label
        hand_y_position = 50  # Starting position for the first hand

        for hand_index, landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks and connections
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect the gesture for each hand
            gesture = detect_gesture(landmarks)

            # Display gesture label
            cv2.putText(frame, f"Hand {hand_index + 1}: {gesture}", (10, hand_y_position), font, 1, (0, 255, 0), 2)
            hand_y_position += gesture_offset_y  # Move the Y position for next hand's gesture

    # Display the number of hands detected, at a fixed location
    cv2.putText(frame, f"Hands Detected: {hand_count}", (10, 30), font, 1, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
