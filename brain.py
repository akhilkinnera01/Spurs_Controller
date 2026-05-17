import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from spatial_math import get_hand_signature
import json
import os

MEMORY_FILE = "gesture_memory.json"


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
        data = {
            "features": [np.asarray(item, dtype=float).tolist() for item in self.X_data],
            "labels": list(self.y_labels),
        }
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print("Brain saved.")

    def load(self):
        if not os.path.exists(MEMORY_FILE):
            return

        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            features = data.get("features", [])
            labels = data.get("labels", [])
            if not isinstance(features, list) or not isinstance(labels, list):
                raise ValueError("gesture memory must contain feature and label lists")
            if len(features) != len(labels):
                raise ValueError("gesture memory feature and label counts differ")

            self.X_data = [np.asarray(item, dtype=float) for item in features]
            self.y_labels = [str(label) for label in labels]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Ignoring invalid gesture memory: {exc}")
            self.X_data = []
            self.y_labels = []
            return

        if len(self.X_data) >= 3:
            self.classifier.fit(self.X_data, self.y_labels)
            self.is_trained = True
        print("Brain reloaded from disk.")
