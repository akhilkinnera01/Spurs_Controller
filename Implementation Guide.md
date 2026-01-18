# Developer Setup & Implementation Guide  
## Gesture Controller

---

## Project Overview

This document outlines the technical environment, dependency constraints, and execution protocols required to run the Gesture Controller.

**Target Architecture:**  
macOS (Apple Silicon M1 / M2 / M3) and Intel Macs  

**Runtime:**  
Python 3.8+

---

## 1. Critical Dependency Note (Apple Silicon)

**Important**

This project explicitly pins:

mediapipe==0.10.5

kotlin
Copy code

### Why this matters

Newer versions of MediaPipe (0.10.9 and above) introduced a breaking change in the macOS ARM64 Python wheels. This breaks the legacy **Solutions API**, specifically:

mp.solutions.hands

csharp
Copy code

The failure typically manifests as:

AttributeError: module 'mediapipe' has no attribute 'solutions'

yaml
Copy code

Rolling MediaPipe back to **0.10.5** restores full access to the hand landmark tensor graph that this application relies on for spatial math and gesture inference.

Do not upgrade this dependency unless the core hand tracking pipeline is refactored away from the Solutions API.

---

## 2. Environment Configuration

To prevent **Path Shadowing** (global Conda or system Python interfering with local imports), the project must run inside a strictly isolated virtual environment.

### Setup Script

Execute the following commands from the project root:

```bash
# 1. Remove any existing virtual environment
rm -rf venv

# 2. Create a fresh virtual environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate
You should now see (venv) prefixed in your terminal prompt.

3. Installation
Ensure requirements.txt exists in the project root.

Install dependencies using the pinned versions:

bash
Copy code
pip install -r requirements.txt
Core Stack
OpenCV (cv2)
Real-time camera capture and vector overlay rendering.

MediaPipe (0.10.5)
21-point hand landmark skeletal tracking.

Scikit-learn
K-Nearest Neighbors (KNN) classifier for gesture inference.

PyAutoGUI
System-level input simulation.

NumPy
High-performance vector arithmetic and Euclidean distance calculations.

4. Execution & Calibration Protocol
Launch
bash
Copy code
python main.py
macOS Permissions
When prompted, grant the following permissions to your Terminal or IDE:

Camera
Required for live video capture.

Accessibility
Required for system volume control via osascript.

Without these permissions, gesture detection may work but system actions will fail silently.

5. Real-Time Training (The “Brain”)
The system starts in an uncalibrated state. You must train the KNN classifier using your own hand geometry.

Each keypress records a feature vector into the model.

Training Steps
Neutral State
Relax your hand.
Press 1 repeatedly while slightly rotating your wrist.

Volume Up
Hold your activation gesture (example: open palm).
Press 2 repeatedly.

Volume Down
Hold your deactivation gesture (example: closed fist).
Press 3 repeatedly.

The model automatically serializes this training data to:

Copy code
gesture_memory.pkl
This allows persistence across restarts.

6. Troubleshooting
Issue
pgsql
Copy code
AttributeError: module 'mediapipe' has no attribute 'solutions'
Causes
Running outside the virtual environment

MediaPipe version mismatch

A local file named mediapipe.py shadowing the package

Stale Python cache files

Fix Checklist
Ensure no file in the project directory is named mediapipe.py

Delete cache files:

bash
Copy code
rm -rf __pycache__
Confirm MediaPipe version:

bash
Copy code
pip show mediapipe
It must report 0.10.5

Recreate the virtual environment using the setup script above

Notes
This project assumes direct camera access and low-latency frame processing. Running inside containers, remote desktops, or sandboxed environments is not supported.

Gesture recognition quality depends heavily on lighting, camera FOV, and user-specific calibration density.

Train generously. Your future self will thank you.

Copy code
