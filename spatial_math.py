import numpy as np

def get_hand_signature(landmarks):
    """
    Converts 21 landmarks into a scale-invariant vector (The "Fingerprint/DNA").
    """
    
    # 1. Convert landmarks to a NumPy matrix (21 points x 2 coordinates)
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # 2. Define the Anchor: The Wrist (Point 0)
    wrist = coords[0]
    
    # 3. Calculate Relative Vectors
    vectors = coords[1:] - wrist
    
    # 4. Calculate Euclidean Distances (Magnitude)
    distances = np.linalg.norm(vectors, axis=1)
    
    # 5. Normalize (The Critical Step)
    max_dist = np.max(distances)
    
    # Safety check to avoid division by zero
    if max_dist < 1e-6:
        return np.zeros_like(distances)
        
    normalized_sig = distances / max_dist
    
    return normalized_sig