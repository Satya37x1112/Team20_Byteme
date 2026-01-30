"""
Blink Detection Module
Uses Eye Aspect Ratio (EAR) to detect natural eye blinks
"""

import numpy as np
from scipy.spatial import distance as dist
from typing import Tuple, Optional
from . import config


class BlinkDetector:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) algorithm.
    Uses dlib 68-point facial landmarks.
    """
    
    # Landmark indices for eyes (dlib 68-point model)
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    
    def __init__(self):
        """Initialize the blink detector."""
        self.blink_count = 0
        self.consecutive_frames = 0
        self.blink_in_progress = False
        self.ear_history = []
        self.smoothing_window = 3
        
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Args:
            eye_landmarks: Array of 6 (x, y) points for one eye
            
        Returns:
            EAR value (lower when eye is closed)
        """
        # Vertical distances
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_both_ears(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate EAR for both eyes.
        
        Args:
            landmarks: Full 68-point facial landmarks array
            
        Returns:
            Tuple of (left_ear, right_ear, average_ear)
        """
        # Extract eye landmarks
        left_eye = np.array([landmarks[i] for i in self.LEFT_EYE_INDICES])
        right_eye = np.array([landmarks[i] for i in self.RIGHT_EYE_INDICES])
        
        # Calculate EAR for each eye
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, avg_ear
    
    def smooth_ear(self, ear: float) -> float:
        """Apply temporal smoothing to reduce noise."""
        self.ear_history.append(ear)
        if len(self.ear_history) > self.smoothing_window:
            self.ear_history.pop(0)
        return np.mean(self.ear_history)
    
    def detect_blink(self, landmarks: np.ndarray) -> dict:
        """
        Detect if a blink occurred.
        
        Args:
            landmarks: 68-point facial landmarks
            
        Returns:
            Dictionary with blink detection results
        """
        result = {
            'blink_detected': False,
            'ear': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'blink_count': self.blink_count,
            'eyes_closed': False
        }
        
        if landmarks is None or len(landmarks) < 68:
            return result
        
        # Calculate EAR
        left_ear, right_ear, avg_ear = self.get_both_ears(landmarks)
        smoothed_ear = self.smooth_ear(avg_ear)
        
        result['ear'] = smoothed_ear
        result['left_ear'] = left_ear
        result['right_ear'] = right_ear
        
        # Check if eyes are closed (below threshold)
        if smoothed_ear < config.EAR_THRESHOLD:
            result['eyes_closed'] = True
            self.consecutive_frames += 1
            
            if self.consecutive_frames >= config.BLINK_CONSEC_FRAMES:
                if not self.blink_in_progress:
                    self.blink_in_progress = True
        else:
            # Eyes opened - check if we were in a blink
            if self.blink_in_progress:
                self.blink_count += 1
                result['blink_detected'] = True
                self.blink_in_progress = False
            
            self.consecutive_frames = 0
        
        result['blink_count'] = self.blink_count
        return result
    
    def reset(self):
        """Reset the blink detector state."""
        self.blink_count = 0
        self.consecutive_frames = 0
        self.blink_in_progress = False
        self.ear_history = []
    
    def get_blink_count(self) -> int:
        """Get current blink count."""
        return self.blink_count
