Project Overview

This document outlines the technical environment, dependency constraints, and execution protocols required to run the Gesture Controller.

Target Architecture: macOS (Apple Silicon M1/M2/M3) & Intel

Runtime: Python 3.8+

1. Critical Dependency Note (Apple Silicon)

Important : > This project explicitly pins mediapipe==0.10.5.

Why? Newer versions of MediaPipe (0.10.9+) introduced a breaking change to the Python Wheels for macOS ARM64 architectures, causing the legacy "Solutions" API (mp.solutions.hands) to fail with an AttributeError. Rolling back to 0.10.5 restores full access to the hand landmark tensor graph required for this application's spatial math logic.

2. Environment Configuration

To prevent "Path Shadowing" (where global Conda installs conflict with local imports), a strictly isolated virtual environment is required.

Setup Script

Execute the following in your terminal to create a clean workspace:

# 1. Nuke any existing environment to ensure a clean slate
rm -rf venv

# 2. Initialize a fresh Python Virtual Environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate


3. Installation

Ensure the requirements.txt file is present in your root directory.

# Install dependencies with the specific pinned versions
pip install -r requirements.txt


Core Stack:

OpenCV (cv2): Real-time frame capture and vector overlay rendering.

MediaPipe (0.10.5): Hand landmark detection (21-point skeletal tracking).

Scikit-Learn: K-Nearest Neighbors (KNN) classifier for gesture inference.

PyAutoGUI: System-level input simulation.

NumPy: High-performance vector arithmetic for Euclidean distance calculations.

4. Execution & Calibration Protocol

Launch

python main.py


Note: You must grant "Camera" and "Accessibility" permissions to your Terminal/IDE when prompted to allow video capture and volume control (osascript).

Real-Time Training (The "Brain")

The system initializes in an uncalibrated state. You must "teach" the KNN model your specific hand variance.

Neutral State: Relax your hand. Press 1 repeatedly while rotating your wrist slightly.

Volume Up: Hold your activation gesture (e.g., Open Palm). Press 2 repeatedly.

Volume Down: Hold your deactivation gesture (e.g., Closed Fist). Press 3 repeatedly.

The model automatically serializes this training data to gesture_memory.pkl for persistence.

5. Troubleshooting

Issue: AttributeError: module 'mediapipe' has no attribute 'solutions'

Fix: You are likely running in a global environment or have a conflicting file named mediapipe.py.

Ensure no file in your folder is named mediapipe.py.

Delete the __pycache__ folder.

Re-run the Setup Script above to force the `venv
