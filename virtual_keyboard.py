import cv2
import numpy as np
import mediapipe as mp
import math
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)

# Keyboard Layout
keys = [['Q','W','E','R','T','Y','U','I','O','P','Bksp'],
        ['A','S','D','F','G','H','J','K','L'],
        ['Z','X','C','V','B','N','M'],
        ['Space']]

def draw_keyboard(img):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            key_w = 100 if key == 'Space' else 60 if key == 'Bksp' else 50
            key_x = j * 60 + 50
            key_y = i * 60 + 100
            cv2.rectangle(img, (key_x, key_y), (key_x + key_w, key_y + 50), (255, 0, 0), -1)
            text_x = key_x + 10 if key != "Space" else key_x + 25
            cv2.putText(img, key, (text_x, key_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img

def get_finger_positions(hand_landmarks, frame):
    h, w, _ = frame.shape
    landmarks = []
    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks.append((cx, cy))
    return landmarks

def is_click(landmarks):
    x1, y1 = landmarks[8]  # Index tip
    x2, y2 = landmarks[4]  # Thumb tip
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist < 30

def is_backspace_gesture(landmarks):
    x1, y1 = landmarks[12]  # Middle tip
    x2, y2 = landmarks[4]   # Thumb tip
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist < 30

def get_key_pressed(x, y):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            key_w = 100 if key == 'Space' else 60 if key == 'Bksp' else 50
            key_x = j * 60 + 50
            key_y = i * 60 + 100
            if key_x < x < key_x + key_w and key_y < y < key_y + 50:
                return key
    return None

# Debounce
last_press_time = 0
typed_text = ""

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    frame = draw_keyboard(frame)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        landmarks = get_finger_positions(hand, frame)

        if len(landmarks) >= 13:
            x, y = landmarks[8]  # Index finger tip

            # Type
            if is_click(landmarks) and time.time() - last_press_time > 0.5:
                key = get_key_pressed(x, y)
                if key:
                    if key == 'Space':
                        typed_text += ' '
                    elif key == 'Bksp':
                        typed_text = typed_text[:-1]
                    else:
                        typed_text += key
                    last_press_time = time.time()
                    cv2.rectangle(frame, (x-25, y-25), (x+25, y+25), (0,255,0), 3)
                    cv2.putText(frame, f"Pressed: {key}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Gesture-based delete
            elif is_backspace_gesture(landmarks) and time.time() - last_press_time > 0.5:
                if typed_text:
                    typed_text = typed_text[:-1]
                last_press_time = time.time()
                cv2.putText(frame, "Deleted", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display typed text
    cv2.putText(frame, typed_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

    cv2.imshow("Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

