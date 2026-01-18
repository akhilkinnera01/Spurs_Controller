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
        self.load() # Auto-load on startup

    def teach(self, landmarks, label):
        signature = get_hand_signature(landmarks)
        self.X_data.append(signature)
        self.y_labels.append(label)
        if len(self.X_data) >= 3:
            self.classifier.fit(self.X_data, self.y_labels)
            self.is_trained = True
            self.save() # Auto-save every time you teach

    def predict(self, landmarks):
        if not self.is_trained: return "Uncertain"
        signature = get_hand_signature(landmarks)
        return self.classifier.predict([signature])[0]

    def save(self):
        # Saves the raw data to a file
        with open("gesture_memory.pkl", "wb") as f:
            pickle.dump((self.X_data, self.y_labels), f)
        print("💾 Brain Saved.")

    def load(self):
        # Loads data if the file exists
        if os.path.exists("gesture_memory.pkl"):
            with open("gesture_memory.pkl", "rb") as f:
                self.X_data, self.y_labels = pickle.load(f)
            
            if len(self.X_data) >= 3:
                self.classifier.fit(self.X_data, self.y_labels)
                self.is_trained = True
            print("Brain Reloaded from Disk.")