# Snack Attack 

## Gesture-Controlled Volume for macOS

This weekend, I wanted to build something fun.

I had a Mac, a webcam, and a simple goal: control my volume without touching the keyboard.

I am a huge Spurs fan. I was watching the game against the Wolves, fourth quarter, tension rising. I was leaned back, mid-snack. Every time the crowd got loud or a big play happened, I wanted to crank the volume. But my hands were greasy and my keyboard deserved better.

So I thought: gestures.

That thought turned into **Snack Attack**, a local, context-aware spatial controller that runs entirely on macOS.

This README explains how it works, why the obvious approach failed, and the specific engineering decisions that made it reliable.

---

## Why the Obvious Approach Failed

My first instinct was pixels.

I used MediaPipe, grabbed the thumb and index finger coordinates, and wrote logic like this:

```python
if thumb_tip.y < index_tip.y:
    print("Volume UP!")
```

It worked perfectly for about five seconds.

Then I leaned back. My hand got smaller in the frame. The distance between my fingers dropped from 100 pixels to 50 pixels. Everything broke.

Move the hand closer, move it farther, rotate it slightly, and the logic collapsed.

That is when it clicked.

This was not a logic problem.

It was a geometry problem.

I needed gesture recognition that worked whether my hand was one foot or ten feet from the camera.

---

## The Architecture

The fix required abandoning pixels and thinking in vectors.

The system ended up with three clean layers:

1. **The Math Engine**
   Converts raw hand landmarks into a scale-invariant fingerprint.

2. **The Brain**
   A k-Nearest Neighbors classifier for instant learning.

3. **The Bridge**
   A macOS-specific volume control path that actually works.

---

## Concept 1: Hand Fingerprints

While the size of your hand changes with distance, the *ratios* of your hand do not.

The math engine builds a signature using this process:

* Anchor the wrist as the origin
* Measure Euclidean distance from the wrist to each landmark
* Normalize all distances by hand scale

Example:

* Hand close to camera: distance = 100, scale = 100 → 1.0
* Hand far away: distance = 50, scale = 50 → 1.0

Distance changes.

The signature does not.

This produces a consistent fingerprint regardless of depth.

### Core Math

```python
import numpy as np

def get_hand_signature(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    wrist = coords[0]
    vectors = coords[1:] - wrist
    distances = np.linalg.norm(vectors, axis=1)
    max_dist = np.max(distances)
    if max_dist < 1e-6:
        return np.zeros_like(distances)
    return distances / max_dist
```

---

## Concept 2: The Brain

With a mathematical fingerprint in hand, the next question was learning.

I could have trained a CNN.

But I did not want to label a thousand images.

I wanted this to work *now*.

So I used **k-Nearest Neighbors**.

k-NN does not generalize. It memorizes.

* Press `2`: this fingerprint means Volume Up
* Press `3`: this fingerprint means Volume Down

That enables instant, one-shot learning.

New gesture. Five examples. Done.

### Brain Implementation

```python
class GestureBrain:
    def __init__(self):
        self.X_data = []
        self.y_labels = []
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.load()

    def teach(self, landmarks, label):
        signature = get_hand_signature(landmarks)
        self.X_data.append(signature)
        self.y_labels.append(label)
        if len(self.X_data) >= 3:
            self.classifier.fit(self.X_data, self.y_labels)
            self.is_trained = True
            self.save()
```

---

## macOS Reality Check

Modern macOS blocks background scripts from controlling media keys.

The model was working. The gestures were recognized. The volume refused to move.

The fix was AppleScript.

By piping commands through `osascript`, the program talks directly to the system volume manager, bypassing UI restrictions.

This is the only reliable way to change volume on modern macOS.

---

## Lessons Learned

### Rotational Invariance

A fist trained straight on is not the same fist sideways.

The fix was not more code.

The fix was more data.

I trained multiple orientations and let k-NN memorize them all.

### Jitter and Hysteresis

Hands flicker between states dozens of times per second.

Mapping that directly to volume creates chaos.

The solution was a buffer.

A gesture must be held for several consecutive frames before it triggers. The latency is small. The stability is massive.

---

## Final Thoughts

The model was never the hard part.

Context was.

* Vector math solved scale
* Hysteresis solved noise
* AppleScript solved macOS

Snack Attack now runs quietly in the background, barely touching the CPU, letting me control volume without lifting a snack-covered finger.

The full source code is on GitHub.

Build your own gesture interface. Just do not trust pixels.
