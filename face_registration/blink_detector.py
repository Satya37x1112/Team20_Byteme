"""
Blink Detection Module using Eye Aspect Ratio (EAR)
Uses dlib facial landmarks for accurate eye tracking
"""

import numpy as np
from scipy.spatial import distance as dist
import dlib
import cv2
from . import config


class BlinkDetector:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) method.
    Uses dlib's 68-point facial landmark predictor.
    """
    
    # Facial landmark indices for eyes (dlib 68-point model)
    LEFT_EYE_INDICES = list(range(42, 48))
    RIGHT_EYE_INDICES = list(range(36, 42))
    
    def __init__(self, predictor_path: str = None):
        """
        Initialize the blink detector.
        
        Args:
            predictor_path: Path to dlib shape predictor file.
                          If None, uses default location.
        """
        self.detector = dlib.get_frontal_face_detector()
        
        # Try to load shape predictor
        if predictor_path is None:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError:
            raise FileNotFoundError(
                f"Shape predictor file not found: {predictor_path}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        # Blink tracking state
        self.ear_threshold = config.EAR_THRESHOLD
        self.consec_frames = config.BLINK_CONSEC_FRAMES
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history = []
        
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_points: Array of 6 (x, y) coordinates for eye landmarks
            
        Returns:
            Eye Aspect Ratio value
        """
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_landmarks(self, gray_frame: np.ndarray, face_rect) -> np.ndarray:
        """
        Get facial landmarks for a detected face.
        
        Args:
            gray_frame: Grayscale image
            face_rect: dlib rectangle of detected face
            
        Returns:
            Array of (x, y) coordinates for all 68 landmarks
        """
        shape = self.predictor(gray_frame, face_rect)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) 
                             for i in range(68)])
        return landmarks
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> tuple:
        """
        Extract eye landmarks from full facial landmarks.
        
        Args:
            landmarks: Full 68-point landmarks array
            
        Returns:
            Tuple of (left_eye, right_eye) landmark arrays
        """
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        return left_eye, right_eye
    
    def detect_blink(self, frame: np.ndarray, face_location: tuple = None) -> dict:
        """
        Detect blink in the current frame.
        
        Args:
            frame: BGR image from camera
            face_location: Optional (top, right, bottom, left) tuple
            
        Returns:
            Dictionary with detection results:
            - 'blink_detected': bool - True if blink occurred this frame
            - 'total_blinks': int - Total blinks counted
            - 'current_ear': float - Current EAR value
            - 'eyes_closed': bool - True if eyes currently closed
            - 'landmarks': array - Facial landmarks if detected
        """
        result = {
            'blink_detected': False,
            'total_blinks': self.blink_counter,
            'current_ear': 0.0,
            'eyes_closed': False,
            'landmarks': None,
            'left_eye': None,
            'right_eye': None
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces or use provided location
        if face_location is not None:
            top, right, bottom, left = face_location
            faces = [dlib.rectangle(left, top, right, bottom)]
        else:
            faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            self.frame_counter = 0
            return result
        
        # Get landmarks for first face
        face = faces[0]
        landmarks = self.get_landmarks(gray, face)
        result['landmarks'] = landmarks
        
        # Get eye landmarks
        left_eye, right_eye = self.get_eye_landmarks(landmarks)
        result['left_eye'] = left_eye
        result['right_eye'] = right_eye
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        result['current_ear'] = ear
        
        # Store EAR history for smoothing
        self.ear_history.append(ear)
        if len(self.ear_history) > 10:
            self.ear_history.pop(0)
        
        # Check for blink
        if ear < self.ear_threshold:
            self.frame_counter += 1
            result['eyes_closed'] = True
        else:
            # Eyes opened after being closed
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                result['blink_detected'] = True
                result['total_blinks'] = self.blink_counter
            
            self.frame_counter = 0
        
        return result
    
    def reset(self):
        """Reset blink counter and state."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history = []
    
    def get_blink_count(self) -> int:
        """Get current blink count."""
        return self.blink_counter
    
    def is_liveness_verified(self) -> bool:
        """Check if minimum blinks have been detected."""
        return self.blink_counter >= config.REQUIRED_BLINKS
