"""
Face Detection and Recognition Utilities
Handles face detection, validation, and encoding extraction
"""

import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional
from . import config


class FaceValidator:
    """
    Validates face for registration:
    - Single face detection
    - Face size validation
    - Face position (inside oval) validation
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize validator with frame dimensions.
        
        Args:
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Calculate oval parameters
        self.oval_center = (
            int(frame_width * config.OVAL_CENTER_X_RATIO),
            int(frame_height * config.OVAL_CENTER_Y_RATIO)
        )
        self.oval_axes = (
            int(frame_width * config.OVAL_WIDTH_RATIO / 2),
            int(frame_height * config.OVAL_HEIGHT_RATIO / 2)
        )
        
        # Calculate acceptable face size range
        oval_area = np.pi * self.oval_axes[0] * self.oval_axes[1]
        self.min_face_area = oval_area * config.MIN_FACE_RATIO
        self.max_face_area = oval_area * config.MAX_FACE_RATIO
        
    def detect_faces(self, frame: np.ndarray, scale: float = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame using face_recognition library.
        
        Args:
            frame: BGR image
            scale: Optional scale factor for processing
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        if scale is None:
            scale = config.PROCESS_FRAME_SCALE
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small, 
            model=config.FACE_DETECTION_MODEL
        )
        
        # Scale back to original size
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((
                int(top / scale),
                int(right / scale),
                int(bottom / scale),
                int(left / scale)
            ))
        
        return scaled_locations
    
    def validate_single_face(self, face_locations: List) -> Tuple[bool, str]:
        """
        Validate that exactly one face is present.
        
        Args:
            face_locations: List of detected face locations
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(face_locations) == 0:
            return False, "No face detected. Please position your face in the frame."
        
        if len(face_locations) > 1:
            return False, "Multiple faces detected. Only one person allowed."
        
        return True, "Single face detected."
    
    def validate_face_size(self, face_location: Tuple[int, int, int, int]) -> Tuple[bool, str]:
        """
        Validate face size is within acceptable range.
        
        Args:
            face_location: (top, right, bottom, left) tuple
            
        Returns:
            Tuple of (is_valid, message)
        """
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        
        if face_area < self.min_face_area:
            return False, "Move closer to the camera."
        
        if face_area > self.max_face_area:
            return False, "Move slightly back."
        
        return True, "Face size OK."
    
    def validate_face_position(self, face_location: Tuple[int, int, int, int]) -> Tuple[bool, str]:
        """
        Validate face is positioned inside the oval frame.
        
        Args:
            face_location: (top, right, bottom, left) tuple
            
        Returns:
            Tuple of (is_valid, message)
        """
        top, right, bottom, left = face_location
        
        # Calculate face center
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
        
        # Check if face center is inside oval
        # Ellipse equation: (x-h)²/a² + (y-k)²/b² <= 1
        oval_x, oval_y = self.oval_center
        a, b = self.oval_axes
        
        # Normalize coordinates
        normalized_x = (face_center_x - oval_x) / a
        normalized_y = (face_center_y - oval_y) / b
        
        # Check if inside ellipse (with some tolerance)
        distance = normalized_x**2 + normalized_y**2
        
        if distance > 0.7:  # Allow some margin
            return False, "Align your face inside the frame."
        
        return True, "Face position OK."
    
    def validate_face(self, frame: np.ndarray) -> dict:
        """
        Perform complete face validation.
        
        Args:
            frame: BGR image
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'message': '',
            'face_location': None,
            'face_count': 0,
            'status': 'detecting'
        }
        
        # Detect faces
        face_locations = self.detect_faces(frame)
        result['face_count'] = len(face_locations)
        
        # Validate single face
        is_single, msg = self.validate_single_face(face_locations)
        if not is_single:
            result['message'] = msg
            result['status'] = 'error' if len(face_locations) > 1 else 'detecting'
            return result
        
        face_location = face_locations[0]
        result['face_location'] = face_location
        
        # Validate position first
        is_positioned, msg = self.validate_face_position(face_location)
        if not is_positioned:
            result['message'] = msg
            result['status'] = 'position'
            return result
        
        # Validate size
        is_sized, msg = self.validate_face_size(face_location)
        if not is_sized:
            result['message'] = msg
            result['status'] = 'size'
            return result
        
        result['valid'] = True
        result['message'] = "Face detected. Please blink naturally."
        result['status'] = 'ready'
        return result
    
    def get_oval_params(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get oval center and axes."""
        return self.oval_center, self.oval_axes


class FaceEncoder:
    """
    Extracts face encodings for storage and comparison.
    Uses face_recognition library with optimized settings.
    """
    
    @staticmethod
    def get_encoding(frame: np.ndarray, face_location: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Extract face encoding from frame.
        
        Args:
            frame: BGR image
            face_location: Optional face location for faster processing
            
        Returns:
            128-dimensional face encoding array, or None if failed
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if face_location is not None:
            face_locations = [face_location]
        else:
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model=config.FACE_DETECTION_MODEL
            )
        
        if len(face_locations) == 0:
            return None
        
        encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations,
            model=config.FACE_ENCODING_MODEL
        )
        
        if len(encodings) == 0:
            return None
        
        return encodings[0]
    
    @staticmethod
    def compare_faces(known_encoding: np.ndarray, unknown_encoding: np.ndarray, tolerance: float = 0.6) -> Tuple[bool, float]:
        """
        Compare two face encodings.
        
        Args:
            known_encoding: Known face encoding
            unknown_encoding: Unknown face encoding to compare
            tolerance: Distance threshold for match
            
        Returns:
            Tuple of (is_match, distance)
        """
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        is_match = distance <= tolerance
        return is_match, distance
