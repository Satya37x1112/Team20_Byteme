"""
Main Application Entry Point
Face Registration System with Controlled Blink Verification
"""

import cv2
import time
import numpy as np
from enum import Enum
from typing import Optional

from . import config
from .blink_detector import BlinkDetector
from .face_utils import FaceValidator, FaceEncoder
from .storage import EmbeddingStorage
from .ui_overlay import UIOverlay


class RegistrationState(Enum):
    """States of the registration process."""
    IDLE = "idle"
    COUNTDOWN = "countdown"
    REGISTERING = "registering"
    AWAITING_BLINK = "awaiting_blink"
    LIVENESS_CONFIRMED = "liveness_confirmed"
    CAPTURING = "capturing"
    NAME_INPUT = "name_input"
    SUCCESS = "success"
    FAILED = "failed"


class FaceRegistrationSystem:
    """
    Main class orchestrating the face registration process.
    Implements secure registration with blink-based liveness detection.
    """
    
    def __init__(self, predictor_path: str = None):
        """
        Initialize the registration system.
        
        Args:
            predictor_path: Path to dlib shape predictor file
        """
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Get actual frame dimensions
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize components
        self.blink_detector = BlinkDetector(predictor_path)
        self.face_validator = FaceValidator(self.frame_width, self.frame_height)
        self.face_encoder = FaceEncoder()
        self.storage = EmbeddingStorage()
        self.ui = UIOverlay(self.frame_width, self.frame_height)
        
        # State management
        self.state = RegistrationState.IDLE
        self.start_time: Optional[float] = None
        self.captured_encoding: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self.failure_reason: str = ""
        self.registered_name: str = ""
        self.current_message: str = ""
        self.message_status: str = "info"
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def reset(self):
        """Reset the registration state."""
        self.state = RegistrationState.IDLE
        self.start_time = None
        self.captured_encoding = None
        self.captured_frame = None
        self.failure_reason = ""
        self.registered_name = ""
        self.current_message = ""
        self.message_status = "info"
        self.blink_detector.reset()
    
    def get_time_remaining(self) -> int:
        """Get remaining registration time in seconds."""
        if self.start_time is None:
            return config.REGISTRATION_TIMEOUT
        
        elapsed = time.time() - self.start_time
        remaining = max(0, config.REGISTRATION_TIMEOUT - int(elapsed))
        return remaining
    
    def start_registration(self):
        """Start the registration process."""
        self.reset()
        self.state = RegistrationState.REGISTERING
        self.start_time = time.time()
        self.blink_detector.reset()
        print("Registration started. Please blink naturally.")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the registration pipeline.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            Processed frame with overlays
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Apply background blur
        display_frame = self.ui.apply_background_blur(frame)
        
        # State machine processing
        if self.state == RegistrationState.IDLE:
            display_frame = self._process_idle(display_frame, frame)
        
        elif self.state == RegistrationState.REGISTERING:
            display_frame = self._process_registering(display_frame, frame)
        
        elif self.state == RegistrationState.LIVENESS_CONFIRMED:
            display_frame = self._process_liveness_confirmed(display_frame, frame)
        
        elif self.state == RegistrationState.SUCCESS:
            display_frame = self.ui.draw_success_screen(display_frame, self.registered_name)
        
        elif self.state == RegistrationState.FAILED:
            display_frame = self.ui.draw_failure_screen(display_frame, self.failure_reason)
        
        # Draw FPS counter (debug)
        self._update_fps()
        cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", 
                   (self.frame_width - 100, self.frame_height - 10),
                   config.FONT, 0.5, (100, 100, 100), 1)
        
        return display_frame
    
    def _process_idle(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process frame in idle state."""
        # Draw oval frame
        display_frame = self.ui.draw_oval_frame(display_frame, 'default')
        
        # Show instructions
        display_frame = self.ui.draw_instructions(display_frame)
        
        # Draw title
        display_frame = self.ui.draw_message(
            display_frame, 
            "Face Registration System", 
            'info', 
            'top'
        )
        
        return display_frame
    
    def _process_registering(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process frame during active registration."""
        # Check timeout
        time_remaining = self.get_time_remaining()
        if time_remaining <= 0:
            self.state = RegistrationState.FAILED
            self.failure_reason = "Registration timeout. Please try again."
            return display_frame
        
        # Draw timer
        display_frame = self.ui.draw_timer(display_frame, time_remaining)
        
        # Validate face
        validation = self.face_validator.validate_face(original_frame)
        
        if not validation['valid']:
            # Draw oval with warning
            oval_status = 'warning' if validation['status'] != 'error' else 'error'
            display_frame = self.ui.draw_oval_frame(display_frame, oval_status)
            
            # Handle multiple faces error
            if validation['face_count'] > 1:
                self.state = RegistrationState.FAILED
                self.failure_reason = validation['message']
                return display_frame
            
            # Show validation message
            display_frame = self.ui.draw_message(
                display_frame, 
                validation['message'],
                'warning' if validation['status'] != 'error' else 'error'
            )
            
            return display_frame
        
        # Face is valid - draw active oval
        display_frame = self.ui.draw_oval_frame(display_frame, 'active')
        
        # Draw face box
        if validation['face_location']:
            display_frame = self.ui.draw_face_box(
                display_frame, 
                validation['face_location'],
                'success'
            )
        
        # Process blink detection
        blink_result = self.blink_detector.detect_blink(original_frame, validation['face_location'])
        
        # Draw blink indicator
        display_frame = self.ui.draw_blink_indicator(
            display_frame,
            blink_result['total_blinks'],
            config.REQUIRED_BLINKS,
            blink_result['eyes_closed']
        )
        
        # Draw eye landmarks for visual feedback
        if blink_result['left_eye'] is not None and blink_result['right_eye'] is not None:
            self._draw_eye_landmarks(display_frame, blink_result['left_eye'], blink_result['right_eye'])
        
        # Check if liveness is verified
        if self.blink_detector.is_liveness_verified():
            self.state = RegistrationState.LIVENESS_CONFIRMED
            self.captured_frame = original_frame.copy()
            
            # Extract face encoding
            self.captured_encoding = self.face_encoder.get_encoding(
                original_frame, 
                validation['face_location']
            )
            
            if self.captured_encoding is None:
                self.state = RegistrationState.FAILED
                self.failure_reason = "Could not extract face encoding. Try again."
            else:
                print("Liveness confirmed! Face captured.")
            
            return display_frame
        
        # Show prompt to blink
        display_frame = self.ui.draw_message(
            display_frame,
            "Liveness check: Please blink naturally",
            'info'
        )
        
        return display_frame
    
    def _process_liveness_confirmed(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process frame after liveness is confirmed."""
        display_frame = self.ui.draw_oval_frame(display_frame, 'success')
        display_frame = self.ui.draw_message(
            display_frame,
            "Liveness confirmed! Enter name in console.",
            'success'
        )
        return display_frame
    
    def _draw_eye_landmarks(self, frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray):
        """Draw eye landmarks for visual feedback."""
        # Draw eye contours
        cv2.polylines(frame, [left_eye.astype(np.int32)], True, config.COLOR_INFO, 1)
        cv2.polylines(frame, [right_eye.astype(np.int32)], True, config.COLOR_INFO, 1)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def prompt_for_name(self) -> Optional[str]:
        """
        Prompt user to enter their name.
        
        Returns:
            Entered name or None if cancelled
        """
        print("\n" + "="*50)
        print("FACE CAPTURED SUCCESSFULLY")
        print("="*50)
        
        while True:
            name = input("Enter your name (or 'cancel' to abort): ").strip()
            
            if name.lower() == 'cancel':
                return None
            
            if len(name) < 2:
                print("Name must be at least 2 characters.")
                continue
            
            if self.storage.exists(name):
                overwrite = input(f"'{name}' already exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != 'y':
                    continue
            
            return name
    
    def save_registration(self, name: str) -> bool:
        """
        Save the captured face encoding with the given name.
        
        Args:
            name: Person's name
            
        Returns:
            True if saved successfully
        """
        if self.captured_encoding is None:
            return False
        
        success = self.storage.register(name, self.captured_encoding)
        
        if success:
            self.registered_name = name
            self.state = RegistrationState.SUCCESS
            print(f"Successfully registered: {name}")
        else:
            self.state = RegistrationState.FAILED
            self.failure_reason = "Failed to save registration."
        
        return success
    
    def run(self):
        """Main application loop."""
        print("\n" + "="*60)
        print("    FACE REGISTRATION SYSTEM WITH BLINK VERIFICATION")
        print("="*60)
        print("\nControls:")
        print("  SPACE  - Start registration")
        print("  R      - Retry after failure")
        print("  Q/ESC  - Quit")
        print("="*60 + "\n")
        
        cv2.namedWindow("Face Registration", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Process frame
                display_frame = self.process_frame(frame)
                
                # Show frame
                cv2.imshow("Face Registration", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                
                elif key == ord(' ') and self.state == RegistrationState.IDLE:
                    self.start_registration()
                
                elif key == ord('r') and self.state == RegistrationState.FAILED:
                    self.reset()
                
                # Handle name input when liveness confirmed
                if self.state == RegistrationState.LIVENESS_CONFIRMED:
                    # Temporarily hide window for console input
                    name = self.prompt_for_name()
                    
                    if name:
                        self.save_registration(name)
                    else:
                        self.state = RegistrationState.FAILED
                        self.failure_reason = "Registration cancelled by user."
                
                # Auto-reset after showing success/failure for a few seconds
                if self.state in [RegistrationState.SUCCESS, RegistrationState.FAILED]:
                    # Wait for keypress to continue
                    pass
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        print("\nRegistration system closed.")
        print(f"Total registered faces: {self.storage.count()}")


def main():
    """Entry point for the face registration system."""
    import sys
    
    # Check for shape predictor file
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    try:
        system = FaceRegistrationSystem(predictor_path)
        system.run()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo download the shape predictor file:")
        print("1. Visit: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("2. Extract the .dat file to the project directory")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
