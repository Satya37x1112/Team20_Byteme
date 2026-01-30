"""
Head Pose Estimator Module
Uses solvePnP with facial landmarks to estimate head orientation (pitch, yaw, roll)
"""

import cv2
import numpy as np
import dlib
from typing import Tuple, Optional
from . import config


class HeadPoseEstimator:
    """
    Estimates head pose using facial landmarks and solvePnP.
    Returns pitch (up/down), yaw (left/right), and roll (tilt).
    """
    
    def __init__(self, predictor_path: str = None):
        """
        Initialize the head pose estimator.
        
        Args:
            predictor_path: Path to dlib shape predictor file
        """
        self.detector = dlib.get_frontal_face_detector()
        
        if predictor_path is None:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError:
            raise FileNotFoundError(
                f"Shape predictor file not found: {predictor_path}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        # 3D model points
        self.model_points = np.array(config.MODEL_POINTS_3D, dtype=np.float64)
        
        # Camera matrix (will be set based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Pose smoothing
        self.pose_history = []
        self.smoothing_window = 3
        
    def _init_camera_matrix(self, frame_width: int, frame_height: int):
        """Initialize camera intrinsic matrix based on frame size."""
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
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
                             for i in range(68)], dtype=np.float64)
        return landmarks
    
    def estimate_pose(self, frame: np.ndarray, face_location: Tuple[int, int, int, int] = None) -> dict:
        """
        Estimate head pose from frame.
        
        Args:
            frame: BGR image
            face_location: Optional (top, right, bottom, left) tuple
            
        Returns:
            Dictionary with pose results:
            - 'success': bool
            - 'pitch': float (up/down, negative=up)
            - 'yaw': float (left/right, negative=left)
            - 'roll': float (tilt)
            - 'landmarks': array
            - 'pose_points': 2D projection points
        """
        result = {
            'success': False,
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'landmarks': None,
            'pose_points': None
        }
        
        h, w = frame.shape[:2]
        
        # Initialize camera matrix if needed
        if self.camera_matrix is None:
            self._init_camera_matrix(w, h)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face or use provided location
        if face_location is not None:
            top, right, bottom, left = face_location
            faces = [dlib.rectangle(left, top, right, bottom)]
        else:
            faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return result
        
        # Get landmarks
        face = faces[0]
        landmarks = self.get_landmarks(gray, face)
        result['landmarks'] = landmarks
        
        # Extract the 6 key points for pose estimation
        # Indices: 30 (nose tip), 8 (chin), 45 (left eye left), 
        #          36 (right eye right), 54 (left mouth), 48 (right mouth)
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[45],  # Left eye left corner
            landmarks[36],  # Right eye right corner
            landmarks[54],  # Left mouth corner
            landmarks[48]   # Right mouth corner
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return result
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Apply smoothing
        smoothed_pitch, smoothed_yaw, smoothed_roll = self._smooth_pose(pitch, yaw, roll)
        
        result['success'] = True
        result['pitch'] = smoothed_pitch
        result['yaw'] = smoothed_yaw
        result['roll'] = smoothed_roll
        
        # Project a 3D axis for visualization
        axis_points = np.array([
            [0, 0, 0],
            [100, 0, 0],   # X axis (red)
            [0, 100, 0],   # Y axis (green)
            [0, 0, 100]    # Z axis (blue)
        ], dtype=np.float64)
        
        projected_points, _ = cv2.projectPoints(
            axis_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        result['pose_points'] = projected_points.reshape(-1, 2)
        
        return result
    
    def estimate_pose_simple(self, landmarks: np.ndarray, frame_width: int) -> Tuple[float, float]:
        """
        Simple landmark-based pose estimation as fallback.
        Uses nose position relative to face center.
        
        Returns:
            Tuple of (yaw, pitch) in degrees
        """
        # Face center (between eyes)
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        face_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        
        # Nose tip
        nose_tip = landmarks[30]
        
        # Calculate yaw from nose offset
        face_width = right_eye_center[0] - left_eye_center[0]
        if face_width > 0:
            nose_offset = (nose_tip[0] - face_center_x) / face_width
            yaw = nose_offset * 45  # Scale to degrees
        else:
            yaw = 0
        
        # Calculate pitch from nose vertical position
        chin = landmarks[8]
        forehead_approx = (landmarks[19] + landmarks[24]) / 2  # Eyebrow centers
        face_height = chin[1] - forehead_approx[1]
        
        if face_height > 0:
            expected_nose_y = forehead_approx[1] + face_height * 0.5
            nose_offset_y = (nose_tip[1] - expected_nose_y) / face_height
            pitch = nose_offset_y * 30  # Scale to degrees
        else:
            pitch = 0
        
        return yaw, pitch
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Decompose rotation matrix
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return pitch, yaw, roll
    
    def _smooth_pose(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """Apply temporal smoothing to reduce jitter."""
        self.pose_history.append((pitch, yaw, roll))
        
        if len(self.pose_history) > self.smoothing_window:
            self.pose_history.pop(0)
        
        if len(self.pose_history) == 0:
            return pitch, yaw, roll
        
        # Average over history
        avg_pitch = np.mean([p[0] for p in self.pose_history])
        avg_yaw = np.mean([p[1] for p in self.pose_history])
        avg_roll = np.mean([p[2] for p in self.pose_history])
        
        return avg_pitch, avg_yaw, avg_roll
    
    def reset(self):
        """Reset pose history."""
        self.pose_history = []
