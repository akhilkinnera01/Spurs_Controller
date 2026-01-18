"Snack Attack" for Mac (And Why It Was Harder Than expected)
This weekend, I wanted to build something fun. I had a Mac, a webcam, and a simple goal: I want to control my volume without touching the keyboard.
I'm a huge Spurs fan, and I was watching the game against the Wolves. The 4th quarter was getting intense. I was leaning back, mid-snack, and every time the crowd got loud or a big play happened, I wanted to increase the volume. But I didn't want to keep reaching over to the keys with greasy hands. I thought, "Gestures would be a great idea, let's make one."
I call it the Snack attack. It's a context-aware spatial controller that runs locally on my Mac.
Here is how I built it, why the "obvious" way failed, and the specific choices that i made, from vector math to kernel hacking, that made it actually work.
1: The Mistake (Pixels)
My first attempt was obvious and first instinct. I installed Google's MediaPipe, got the coordinates of my thumb and index finger, and wrote a script that looked like this:
# My Logic
if thumb_tip.y < index_tip.y:
    print("Volume UP!")
It worked perfectly… for exactly 5 seconds.
The moment I leaned back, it stopped and broke. Why? Because my hand got smaller. The distance between my fingers changed from 100 pixels to 50 pixels, breaking my code. If I moved my hand to the corner of the screen, the coordinates shifted, and the logic failed again.
I realized it's not a Logic problem rather a Geometry problem. I needed a way to recognize a "Fist" or "Open Palm" that worked whether I was 1 foot or 10 feet away from the camera.
2: The Architecture
To fix this, I had to stop thinking about pixels and start thinking about vectors, I googled and the solution: 3-layer architecture:
The Math Engine: Converts raw hand data into a "Fingerprint or DNA"
The Brain: k-Nearest Neighbors (k-NN) for One-Shot Learning.
The Bridge: Bypasses macOS security to control hardware.

Concept 1: Fingerprint/DNA
I realized that while the size of my hand changes based on distance, the ratios of my hand do not change.
I wrote a normalization engine that does this:
Anchor: Treat the Wrist as point (0,0).
Measure: Calculate the Euclidean distance from the Wrist to every fingertip.
Normalize: Divide every distance by the Hand Scale which would be the distance from Wrist to Middle Finger.

$$\text{Normalized Value} = \frac{\text{Distance to Joint}}{\text{Total Hand Size}}$$
Example:
Hand close to camera: Raw Distance = 100px. Hand Size = 100px. Result = 1.0.
Hand far away: Raw Distance = 50px. Hand Size = 50px. Result = 1.0.

So, now matter how far i was, the signature was identical.
Concept 2: k-NN
With the mathematical fingerprint/DNA (a list of 20 numbers), we need the the computer to understand it.
I could have trained a Convolutional Neural Network (CNN). But honestly? I didn't want to spend 4 hours labeling 1,000 images. I wanted to use this now.
I went with k-Nearest Neighbors (k-NN). It Mimics things. It doesn't look for patterns; it just memorizes what you show it.
I press '2', and it memorizes: "This list of numbers is 'Volume Up'."
I press '3', and it memorizes: "This list is 'Volume Down'."

This enabled One-Shot Learning. We can show it a new gesture (like a Spiderman web-shooter pose), press a button 5 times, and the system learns and remembers it instantly. No GPU training required.
3: The Build (How to do it)
If you want to build this yourself, here is the exact roadmap/code.
Step 1: The Math (spatial_math.py)
This is the core part. This takes the MediaPipe landmarks and turns them into our "Fingerprint/DNA."
import numpy as np
def get_hand_signature(landmarks):
    # Convert to NumPy array
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # 1. Anchor (Wrist)
    wrist = coords[0]
    
    # 2. Calculating Relative Vectors (Distance from Wrist)
    vectors = coords[1:] - wrist
    distances = np.linalg.norm(vectors, axis=1)
    
    # 3. Normalizing (Divide by max distance)
    # Standardizing the Scale
    max_dist = np.max(distances)
    if max_dist < 1e-6: return np.zeros_like(distances)
    
    return distances / max_dist
Step 2: The Brain (brain.py)
The Brain handles learning. It uses pickle to persist your hand data so it remembers your gestures even after you restart the script.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from spatial_math import get_hand_signature
import pickle
import os

class GestureBrain:
    def __init__(self):
        self.X_data = []
        self.y_labels = []
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.load() #Auto-load on startup

    def teach(self, landmarks, label):
        signature = get_hand_signature(landmarks)
        self.X_data.append(signature)
        self.y_labels.append(label)
        if len(self.X_data) >= 3:
            self.classifier.fit(self.X_data, self.y_labels)
            self.is_trained = True
            self.save() #Auto-save every time you teach

    def predict(self, landmarks):
        if not self.is_trained: return "Uncertain"
        signature = get_hand_signature(landmarks)
        return self.classifier.predict([signature])[0]

    def save(self):
        # Saves the raw data to a file
        with open("gesture_memory.pkl", "wb") as f:
            pickle.dump((self.X_data, self.y_labels), f)
        print("Brain Saved.")

    def load(self):
        # Loads data if the file exists
        if os.path.exists("gesture_memory.pkl"):
            with open("gesture_memory.pkl", "rb") as f:
                self.X_data, self.y_labels = pickle.load(f)
            
            if len(self.X_data) >= 3:
                self.classifier.fit(self.X_data, self.y_labels)
                self.is_trained = True
            print("Brain Reloaded from Disk.")
Step 3: macOS (main.py)
Modern macOS (Sequoia/Sonoma) blocks background scripts from controlling media keys. The Python script was detecting the gesture, but the volume wouldn't change.
I found online that you can use AppleScript (osascript) to bypass this and talk directly to the system's volume settings. This script brings everything together, using subprocess to trigger AppleScript commands for reliable volume control.
import cv2
import mediapipe as mp
import pyautogui
import time
import subprocess
from brain import GestureBrain

# Lower this if it feels slow, increase if it triggers too easily
CONFIDENCE_THRESHOLD = 8  # Frames to hold gesture before it starts

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
        
        # 2. TRIGGER ACTION
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
    cv2.imshow("Mac Gesture", frame)

cap.release()
cv2.destroyAllWindows()
By piping commands directly into the system event manager, I finally did it.
Part 4: Lessons Learned
Even with the code working, I ran into two issues that I had to solve
1. Rotational Invariance
I trained the model with a "Fist" (knuckles forward). But when I sat back, I held my fist sideways, the model didn't recognize it.
The Fix: I didn't change the code. I changed the Data. I trained it by rotating my fist saying that looks like Shape A, Shape B, AND Shape C. Since k-NN just memorizes, it accepted all three variations as the same command.

2. The Jitter Problem (Hysteresis)
Computer vision is noisy. I read online that human hand flickers between states 30 times a second. If I mapped this directly to volume, my speakers would go crazy.
Fix: I added a Buffer. The system requires a gesture to be held for 5 consecutive frames before it starts. This adds a tiny bit of latency (100ms) so that it feels "solid" rather than "glitchy."

Final Thoughts
The final result is good. I have a script running in the background that uses almost no CPU. I can use my hand to pause videos or make a fist to mute my mic in a meeting.
The biggest takeaway? The hard part of AI isn't the model. It's the context. The model knew where my hand was, but it took vector math to handle the scale, hysteresis logic to handle the jitter, and Apple script to actually make it come to life.
You can find the full source code on my GitHub. Go build your own gesture based interface.
