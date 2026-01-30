"""
Configuration settings for Face Registration System
"""

# Timer Settings
REGISTRATION_TIMEOUT = 16  # seconds

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
PROCESS_FRAME_SCALE = 0.5  # Scale for processing (performance optimization)

# Oval Frame Settings
OVAL_CENTER_X_RATIO = 0.5  # Center of frame
OVAL_CENTER_Y_RATIO = 0.45  # Slightly above center
OVAL_WIDTH_RATIO = 0.35  # Relative to frame width
OVAL_HEIGHT_RATIO = 0.55  # Relative to frame height

# Face Size Validation (relative to oval size)
MIN_FACE_RATIO = 0.4  # Face must be at least 40% of oval
MAX_FACE_RATIO = 0.95  # Face must not exceed 95% of oval

# Blink Detection Settings
EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold for blink
BLINK_CONSEC_FRAMES = 2  # Consecutive frames below threshold to count as blink
REQUIRED_BLINKS = 1  # Minimum blinks required for liveness

# Face Recognition Settings
FACE_ENCODING_MODEL = "small"  # "small" for speed, "large" for accuracy
FACE_DETECTION_MODEL = "hog"  # "hog" for CPU, "cnn" for GPU

# Storage Settings
EMBEDDINGS_FILE = "face_embeddings.pkl"

# UI Colors (BGR format)
COLOR_SUCCESS = (0, 200, 0)  # Green
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_ERROR = (0, 0, 255)  # Red
COLOR_INFO = (255, 200, 0)  # Cyan
COLOR_OVAL = (200, 200, 200)  # Light gray
COLOR_OVAL_ACTIVE = (0, 255, 0)  # Green when face aligned
COLOR_TEXT_BG = (0, 0, 0)  # Black background for text

# Font Settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
