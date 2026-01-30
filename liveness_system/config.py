"""
Configuration settings for Random Challenge Liveness Verification System
With Two-Stage Verification: Blink Detection + Head Pose Challenge
"""
import random

# Timer Settings
VERIFICATION_TIMEOUT = 20  # seconds total
BLINK_PHASE_TIMEOUT = 10  # seconds for blink verification
POSE_PHASE_TIMEOUT = 10  # seconds for head pose challenge

# Blink Detection Settings
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink detection
BLINK_CONSEC_FRAMES = 2  # Consecutive frames below threshold to count as blink
REQUIRED_BLINKS = 3  # Number of blinks required to pass

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
PROCESS_FRAME_SCALE = 0.5  # Scale for face detection

# Performance Optimization
FACE_DETECTION_SKIP_FRAMES = 2
POSE_DETECTION_SKIP_FRAMES = 1

# Oval Frame Settings
OVAL_CENTER_X_RATIO = 0.5
OVAL_CENTER_Y_RATIO = 0.45
OVAL_WIDTH_RATIO = 0.38
OVAL_HEIGHT_RATIO = 0.58

# Face Size Validation
MIN_FACE_RATIO = 0.35
MAX_FACE_RATIO = 0.95

# Head Pose Thresholds (degrees) with slight randomness for anti-replay
def get_pose_thresholds():
    """Generate thresholds with slight randomness to prevent replay attacks."""
    base_yaw = 10  # Base threshold for LEFT/RIGHT (lowered for sensitivity)
    base_pitch = 8  # Base threshold for UP/DOWN (lowered for sensitivity)
    
    # Add slight randomness (±2 degrees)
    yaw_threshold = base_yaw + random.uniform(-2, 2)
    pitch_threshold = base_pitch + random.uniform(-2, 2)
    
    return {
        'LEFT': yaw_threshold,        # Positive yaw = head turned left (from camera view)
        'RIGHT': -yaw_threshold,      # Negative yaw = head turned right (from camera view)
        'UP': -pitch_threshold,       # Negative pitch
        'DOWN': pitch_threshold       # Positive pitch
    }

# Challenge Settings
CHALLENGES = ["LEFT", "RIGHT", "UP", "DOWN"]
CHALLENGE_DISPLAY_NAMES = {
    "LEFT": "Turn your head LEFT",
    "RIGHT": "Turn your head RIGHT",
    "UP": "Look UP",
    "DOWN": "Look DOWN"
}

# Movement stability - require consistent pose for N frames
POSE_STABILITY_FRAMES = 3  # Reduced for faster response

# Face Recognition Settings
FACE_ENCODING_MODEL = "small"
FACE_DETECTION_MODEL = "hog"
RECOGNITION_TOLERANCE = 0.6

# Storage Settings
EMBEDDINGS_FILE = "face_embeddings.pkl"

# UI Colors (BGR format)
COLOR_SUCCESS = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_INFO = (255, 200, 0)
COLOR_CHALLENGE = (0, 255, 255)  # Yellow for challenge text
COLOR_BLINK = (255, 0, 255)  # Magenta for blink phase
COLOR_OVAL = (200, 200, 200)
COLOR_OVAL_ACTIVE = (0, 255, 0)
COLOR_TEXT_BG = (0, 0, 0)

# Font Settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# 3D Model Points for Head Pose Estimation (generic face model)
# These points correspond to specific facial landmarks
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),             # Nose tip (landmark 30)
    (0.0, -330.0, -65.0),        # Chin (landmark 8)
    (-225.0, 170.0, -135.0),     # Left eye left corner (landmark 45)
    (225.0, 170.0, -135.0),      # Right eye right corner (landmark 36)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 54)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 48)
]

# Landmark indices for pose estimation (dlib 68-point model)
POSE_LANDMARK_INDICES = [30, 8, 45, 36, 54, 48]
