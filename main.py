import cv2
import mediapipe as mp
import pyautogui
import time
import subprocess
from brain import GestureBrain

# --- SETTINGS ---
# Lower this if it feels slow, raise if it triggers too easily
CONFIDENCE_THRESHOLD = 8  # Frames to hold gesture before firing

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

brain = GestureBrain()
cap = cv2.VideoCapture(0)

def change_volume(amount):
    """Reliably changes volume on macOS and prints status."""
    try:
        # Get current volume (0-100)
        cmd_get = ["osascript", "-e", "output volume of (get volume settings)"]
        result = subprocess.run(cmd_get, capture_output=True, text=True, check=True)
        current_vol = int(result.stdout.strip())
        
        # Calculate new volume
        new_vol = max(0, min(100, current_vol + amount))
        
        # Set new volume
        cmd_set = ["osascript", "-e", f"set volume output volume {new_vol}"]
        subprocess.run(cmd_set, check=True)
        
        print(f"Volume changed: {current_vol} -> {new_vol}")
        return True
    except Exception as e:
        print(f"Volume Control Error: {e}")
        return False

# State Variables
current_state = "Neutral"
frame_counter = 0
last_fired_time = time.time()

print("System Online.")
print("INSTRUCTIONS:")
print("  [1] Teach 'Neutral' (Relaxed Hand)")
print("  [2] Teach 'Volume UP' (e.g., Open Palm)")
print("  [3] Teach 'Volume DOWN' (e.g., Fist)")
print("  [Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for mirror effect (easier to use)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    prediction = "Waiting for data..."
    
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0].landmark
        
        # Draw Skeleton
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        # 1. PREDICT
        prediction = brain.predict(lms)
        
        # 2. TRIGGER ACTION (The "Intent" Engine)
        if prediction == current_state and prediction != "Neutral":
            frame_counter += 1
        else:
            frame_counter = 0
            current_state = prediction
            
        # Fire Action if held for X frames
        if frame_counter > CONFIDENCE_THRESHOLD and (time.time() - last_fired_time > 0.3):
            if prediction == "Volume_Up":
                change_volume(10)
            elif prediction == "Volume_Down":
                change_volume(-10)
            
            last_fired_time = time.time()
            frame_counter = 0 # Reset

    # 3. TEACHING
    key = cv2.waitKey(1)
    if key == ord('1'): 
        brain.teach(lms, "Neutral")
        print("Saved: Neutral")
    elif key == ord('2'): 
        brain.teach(lms, "Volume_Up")
        print("Saved: Volume_Up")
    elif key == ord('3'): 
        brain.teach(lms, "Volume_Down")
        print("Saved: Volume_Down")
    elif key == ord('q'):
        break

    # UI Overlay
    cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
    cv2.putText(frame, f"State: {prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Mac Gesture Controller", frame)

cap.release()
cv2.destroyAllWindows()