import cv2
import numpy as np
import pyttsx3
import time
from collections import deque

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Helper to count fingers based on contours
def count_fingers(contour, hull, defects, frame):
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

        if angle <= np.pi/2 and d > 10000:
            count += 1
            cv2.circle(frame, far, 8, [211, 84, 0], -1)
    return count

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_gesture = None
gesture_queue = deque(maxlen=20)  # For stabilizing gestures
last_spoken = ''
last_time_spoken = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0,255,0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(contour)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)

        fingers = count_fingers(contour, hull, defects, roi) + 1

        # Stabilize gesture with moving average
        gesture_queue.append(fingers)
        avg_fingers = round(sum(gesture_queue) / len(gesture_queue))

        gesture = None
        if avg_fingers == 1:
            gesture = "One"
        elif avg_fingers == 2:
            gesture = "Victory ✌️"
        elif avg_fingers == 3:
            gesture = "Three"
        elif avg_fingers == 4:
            gesture = "Four"
        elif avg_fingers == 5:
            gesture = "Five / Open Hand"
        elif avg_fingers == 0:
            gesture = "Fist / Thumbs Up? 👍"

        if gesture != prev_gesture:
            prev_gesture = gesture
            now = time.time()
            if now - last_time_spoken > 1.5:
                last_time_spoken = now
                if gesture:
                    print("Gesture Detected:", gesture)
                    speak(gesture)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()