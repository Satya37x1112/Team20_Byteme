"""
Main Application Entry Point
Two-Stage Liveness Verification System:
  Stage 1: Blink Detection (verify live eyes)
  Stage 2: Random Head Pose Challenge (LEFT/RIGHT/UP/DOWN)
"""

import cv2
import time
import numpy as np
from enum import Enum
from typing import Optional

from . import config
from .blink_detector import BlinkDetector
from .challenge_generator import ChallengeGenerator
from .head_pose_estimator import HeadPoseEstimator
from .face_utils import FaceValidator, FaceEncoder
from .storage import EmbeddingStorage
from .ui_overlay import UIOverlay


class VerificationState(Enum):
    """States of the verification process."""
    IDLE = "idle"
    BLINK_PHASE = "blink_phase"  # Stage 1: Blink verification
    POSE_PHASE = "pose_phase"    # Stage 2: Head pose challenge
    NAME_INPUT = "name_input"
    SUCCESS = "success"
    FAILED = "failed"
    RECOGNIZED = "recognized"


class LivenessVerificationSystem:
    """
    Two-stage liveness verification system:
    1. Blink detection to verify live eyes
    2. Random head pose challenge to prevent video replay
    """
    
    def __init__(self, predictor_path: str = None):
        """Initialize the verification system."""
        # Initialize camera
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize components
        self.blink_detector = BlinkDetector()
        self.challenge_generator = ChallengeGenerator()
        self.pose_estimator = HeadPoseEstimator(predictor_path)
        self.face_validator = FaceValidator(self.frame_width, self.frame_height)
        self.face_encoder = FaceEncoder()
        self.storage = EmbeddingStorage()
        self.ui = UIOverlay(self.frame_width, self.frame_height)
        
        # State management
        self.state = VerificationState.IDLE
        self.phase_start_time: Optional[float] = None
        self.blink_phase_start: Optional[float] = None
        self.pose_phase_start: Optional[float] = None
        self.captured_encoding: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self.failure_reason: str = ""
        self.registered_name: str = ""
        self.recognized_name: str = ""
        self.input_name: str = ""
        
        # Challenge state
        self.current_challenge = None
        self.challenge_display = None
        self.challenge_progress = 0
        self.pose_stability_counter = 0
        
        # Performance optimization
        self.frame_count = 0
        self.last_face_location = None
        self.last_validation = None
        self.last_pose = {'yaw': 0, 'pitch': 0}
        self.last_ear = 0.3
        
        # Recognition
        self.recognition_cooldown = 0
        self.recognized_display_start = 0
        self.recognition_match_count = 0
        self.last_recognized_name = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def reset(self):
        """Reset the verification state."""
        self.state = VerificationState.IDLE
        self.phase_start_time = None
        self.blink_phase_start = None
        self.pose_phase_start = None
        self.captured_encoding = None
        self.captured_frame = None
        self.failure_reason = ""
        self.registered_name = ""
        self.recognized_name = ""
        self.input_name = ""
        self.current_challenge = None
        self.challenge_display = None
        self.challenge_progress = 0
        self.pose_stability_counter = 0
        self.last_face_location = None
        self.last_validation = None
        self.last_pose = {'yaw': 0, 'pitch': 0}
        self.last_ear = 0.3
        self.blink_detector.reset()
        self.challenge_generator.reset()
        self.pose_estimator.reset()
        self.recognition_match_count = 0
        self.last_recognized_name = None
    
    def get_blink_time_remaining(self) -> int:
        """Get remaining time for blink phase."""
        if self.blink_phase_start is None:
            return config.BLINK_PHASE_TIMEOUT
        elapsed = time.time() - self.blink_phase_start
        return max(0, config.BLINK_PHASE_TIMEOUT - int(elapsed))
    
    def get_pose_time_remaining(self) -> int:
        """Get remaining time for pose phase."""
        if self.pose_phase_start is None:
            return config.POSE_PHASE_TIMEOUT
        elapsed = time.time() - self.pose_phase_start
        return max(0, config.POSE_PHASE_TIMEOUT - int(elapsed))
    
    def start_verification(self):
        """Start the two-stage verification process."""
        self.reset()
        self.state = VerificationState.BLINK_PHASE
        self.blink_phase_start = time.time()
        print("Stage 1: Blink Verification Started")
    
    def start_pose_phase(self):
        """Transition to head pose challenge phase."""
        self.state = VerificationState.POSE_PHASE
        self.pose_phase_start = time.time()
        
        # Generate random challenge AFTER blink verification (anti-replay)
        self.current_challenge, self.challenge_display, _ = self.challenge_generator.generate_challenge()
        print(f"Stage 2: Head Pose Challenge - {self.challenge_display}")
    
    def check_for_known_face(self, frame: np.ndarray) -> Optional[str]:
        """Check if current face matches any registered face."""
        if self.storage.count() == 0 or self.last_face_location is None:
            return None
        
        if self.frame_count % 15 != 0:
            return None
        
        encoding = self.face_encoder.get_encoding(frame, self.last_face_location)
        if encoding is not None:
            return self.storage.find_match(encoding, config.RECOGNITION_TOLERANCE)
        return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the verification pipeline."""
        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        
        # Apply background blur for non-final screens
        if self.state in [VerificationState.SUCCESS, VerificationState.FAILED, 
                          VerificationState.RECOGNIZED]:
            display_frame = frame.copy()
        else:
            display_frame = self.ui.apply_background_blur(frame)
        
        # State machine
        if self.state == VerificationState.IDLE:
            display_frame = self._process_idle(display_frame, frame)
        elif self.state == VerificationState.BLINK_PHASE:
            display_frame = self._process_blink_phase(display_frame, frame)
        elif self.state == VerificationState.POSE_PHASE:
            display_frame = self._process_pose_phase(display_frame, frame)
        elif self.state == VerificationState.NAME_INPUT:
            display_frame = self._process_name_input(display_frame)
        elif self.state == VerificationState.SUCCESS:
            display_frame = self.ui.draw_success_screen(display_frame, self.registered_name)
        elif self.state == VerificationState.FAILED:
            display_frame = self.ui.draw_failure_screen(display_frame, self.failure_reason)
        elif self.state == VerificationState.RECOGNIZED:
            display_frame = self._process_recognized(display_frame)
        
        # FPS display
        self._update_fps()
        cv2.putText(display_frame, f"FPS: {self.current_fps:.0f}", 
                   (self.frame_width - 70, self.frame_height - 10),
                   config.FONT, 0.4, (100, 100, 100), 1)
        
        return display_frame
    
    def _process_idle(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process idle state with face recognition."""
        display_frame = self.ui.draw_oval_frame(display_frame, 'default')
        
        # Detect face
        if self.frame_count % config.FACE_DETECTION_SKIP_FRAMES == 0:
            self.last_validation = self.face_validator.validate_face(original_frame)
            if self.last_validation and self.last_validation['face_location']:
                self.last_face_location = self.last_validation['face_location']
        
        # Check for known face
        if self.storage.count() > 0 and self.recognition_cooldown <= 0:
            recognized = self.check_for_known_face(original_frame)
            if recognized:
                if recognized == self.last_recognized_name:
                    self.recognition_match_count += 1
                else:
                    self.recognition_match_count = 1
                    self.last_recognized_name = recognized
                
                if self.recognition_match_count >= 2:
                    self.recognized_name = recognized
                    self.state = VerificationState.RECOGNIZED
                    self.recognized_display_start = time.time()
                    self.recognition_match_count = 0
                    return display_frame
            else:
                self.recognition_match_count = 0
                self.last_recognized_name = None
        
        if self.recognition_cooldown > 0:
            self.recognition_cooldown -= 1
        
        display_frame = self.ui.draw_instructions(display_frame)
        display_frame = self.ui.draw_message(display_frame, "Two-Stage Liveness Verification", 'info', 'top')
        
        count = self.storage.count()
        if count > 0:
            cv2.putText(display_frame, f"Registered: {count}", 
                       (20, self.frame_height - 10),
                       config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return display_frame
    
    def _process_recognized(self, display_frame: np.ndarray) -> np.ndarray:
        """Show recognized person screen."""
        display_frame = self.ui.draw_recognized_screen(display_frame, self.recognized_name)
        
        if time.time() - self.recognized_display_start > 5:
            self.state = VerificationState.IDLE
            self.recognition_cooldown = 300
            self.recognized_name = ""
            self.recognition_match_count = 0
            self.last_recognized_name = None
        
        return display_frame
    
    def _process_blink_phase(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process Stage 1: Blink verification."""
        # Check timeout
        time_remaining = self.get_blink_time_remaining()
        if time_remaining <= 0:
            self.state = VerificationState.FAILED
            self.failure_reason = "Blink verification timeout."
            return display_frame
        
        # Draw phase indicator and timer
        display_frame = self.ui.draw_phase_indicator(display_frame, 0)
        display_frame = self.ui.draw_timer(display_frame, time_remaining)
        
        # Validate face
        if self.frame_count % config.FACE_DETECTION_SKIP_FRAMES == 0:
            self.last_validation = self.face_validator.validate_face(original_frame)
        
        validation = self.last_validation or {
            'valid': False, 'message': 'Detecting face...', 
            'status': 'detecting', 'face_location': None, 'face_count': 0
        }
        
        if not validation['valid']:
            oval_status = 'warning' if validation['status'] != 'error' else 'error'
            display_frame = self.ui.draw_oval_frame(display_frame, oval_status)
            
            if validation['face_count'] > 1:
                self.state = VerificationState.FAILED
                self.failure_reason = "Multiple faces detected."
                return display_frame
            
            display_frame = self.ui.draw_blink_counter(
                display_frame, 
                self.blink_detector.get_blink_count(),
                config.REQUIRED_BLINKS,
                self.last_ear
            )
            display_frame = self.ui.draw_message(display_frame, validation['message'], 'warning')
            return display_frame
        
        # Face valid
        display_frame = self.ui.draw_oval_frame(display_frame, 'active')
        self.last_face_location = validation['face_location']
        
        # Get landmarks for blink detection
        pose_result = self.pose_estimator.estimate_pose(original_frame, validation['face_location'])
        
        if pose_result['success'] and pose_result['landmarks'] is not None:
            # Detect blinks
            blink_result = self.blink_detector.detect_blink(pose_result['landmarks'])
            self.last_ear = blink_result['ear']
            
            # Draw blink counter
            display_frame = self.ui.draw_blink_counter(
                display_frame, 
                blink_result['blink_count'],
                config.REQUIRED_BLINKS,
                blink_result['ear']
            )
            
            # Check if enough blinks
            if blink_result['blink_count'] >= config.REQUIRED_BLINKS:
                print(f"Blink verification passed! ({blink_result['blink_count']} blinks)")
                self.start_pose_phase()
                return display_frame
            
            # Visual feedback for blink
            if blink_result['eyes_closed']:
                display_frame = self.ui.draw_message(display_frame, "Eyes closed...", 'info')
            elif blink_result['blink_detected']:
                display_frame = self.ui.draw_message(display_frame, "Blink detected!", 'success')
        else:
            display_frame = self.ui.draw_blink_counter(
                display_frame, 
                self.blink_detector.get_blink_count(),
                config.REQUIRED_BLINKS,
                self.last_ear
            )
        
        return display_frame
    
    def _process_pose_phase(self, display_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Process Stage 2: Head pose challenge."""
        # Check timeout
        time_remaining = self.get_pose_time_remaining()
        if time_remaining <= 0:
            self.state = VerificationState.FAILED
            self.failure_reason = "Head pose challenge timeout."
            return display_frame
        
        # Draw phase indicator and timer
        display_frame = self.ui.draw_phase_indicator(display_frame, 1)
        display_frame = self.ui.draw_timer(display_frame, time_remaining)
        
        # Validate face
        if self.frame_count % config.FACE_DETECTION_SKIP_FRAMES == 0:
            self.last_validation = self.face_validator.validate_face(original_frame)
        
        validation = self.last_validation or {
            'valid': False, 'message': 'Detecting face...', 
            'status': 'detecting', 'face_location': None, 'face_count': 0
        }
        
        if not validation['valid']:
            oval_status = 'warning' if validation['status'] != 'error' else 'error'
            display_frame = self.ui.draw_oval_frame(display_frame, oval_status)
            
            if validation['face_count'] > 1:
                self.state = VerificationState.FAILED
                self.failure_reason = "Multiple faces detected."
                return display_frame
            
            display_frame = self.ui.draw_message(
                display_frame, validation['message'],
                'warning' if validation['status'] != 'error' else 'error'
            )
            
            if self.challenge_display:
                display_frame = self.ui.draw_challenge(display_frame, self.challenge_display, 0)
            
            return display_frame
        
        # Face valid
        display_frame = self.ui.draw_oval_frame(display_frame, 'challenge')
        self.last_face_location = validation['face_location']
        
        # Draw face box
        if validation['face_location']:
            display_frame = self.ui.draw_face_box(display_frame, validation['face_location'], 'success')
        
        # Estimate head pose
        pose_result = self.pose_estimator.estimate_pose(original_frame, validation['face_location'])
        
        if pose_result['success'] and pose_result['landmarks'] is not None:
            # Use simple landmark-based yaw for more reliable LEFT/RIGHT detection
            simple_yaw, simple_pitch = self.pose_estimator.estimate_pose_simple(
                pose_result['landmarks'], self.frame_width
            )
            
            # Combine: use simple method for yaw (more reliable for left/right)
            # Use solvePnP for pitch (more accurate for up/down)
            final_yaw = simple_yaw
            final_pitch = pose_result['pitch']
            
            self.last_pose['yaw'] = final_yaw
            self.last_pose['pitch'] = final_pitch
            
            # Draw pose indicator
            display_frame = self.ui.draw_pose_indicator(
                display_frame, 
                final_yaw, 
                final_pitch,
                self.current_challenge
            )
            
            # Check challenge progress
            progress, feedback = self.challenge_generator.get_progress_feedback(
                final_yaw, final_pitch
            )
            self.challenge_progress = progress
            
            # Check if challenge completed
            if self.challenge_generator.check_challenge_completed(
                final_yaw, final_pitch
            ):
                self.pose_stability_counter += 1
                
                if self.pose_stability_counter >= config.POSE_STABILITY_FRAMES:
                    # Both stages completed!
                    self.captured_frame = original_frame.copy()
                    self.captured_encoding = self.face_encoder.get_encoding(
                        original_frame, validation['face_location']
                    )
                    
                    if self.captured_encoding is None:
                        self.state = VerificationState.FAILED
                        self.failure_reason = "Could not extract face encoding."
                    else:
                        self.state = VerificationState.NAME_INPUT
                        self.input_name = ""
                        print("Both stages completed! Liveness verified.")
                    
                    return display_frame
            else:
                self.pose_stability_counter = 0
            
            if feedback:
                display_frame = self.ui.draw_message(display_frame, feedback, 'warning')
        
        # Draw challenge
        if self.challenge_display:
            display_frame = self.ui.draw_challenge(
                display_frame, self.challenge_display, self.challenge_progress
            )
        
        return display_frame
    
    def _process_name_input(self, display_frame: np.ndarray) -> np.ndarray:
        """Show name input screen."""
        return self.ui.draw_name_input_screen(display_frame, self.input_name)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def handle_key_input(self, key: int) -> bool:
        """Handle keyboard input. Returns True if should quit."""
        if key == -1:
            return False
        
        key = key & 0xFF
        
        if key == ord('q') or key == 27:
            return True
        
        if key == ord(' ') and self.state == VerificationState.IDLE:
            self.start_verification()
        
        if key == ord('r') and self.state in [VerificationState.FAILED, VerificationState.SUCCESS]:
            self.reset()
        
        if key == ord(' ') and self.state == VerificationState.RECOGNIZED:
            self.state = VerificationState.IDLE
            self.recognition_cooldown = 300
        
        # Name input handling
        if self.state == VerificationState.NAME_INPUT:
            if key == 13 and len(self.input_name) >= 2:  # Enter
                if self.storage.register(self.input_name, self.captured_encoding):
                    self.registered_name = self.input_name
                    self.state = VerificationState.SUCCESS
                    print(f"Registered: {self.input_name}")
                else:
                    self.state = VerificationState.FAILED
                    self.failure_reason = "Failed to save."
            elif key == 8:  # Backspace
                self.input_name = self.input_name[:-1]
            elif key == 27:  # ESC
                self.state = VerificationState.FAILED
                self.failure_reason = "Cancelled by user."
            elif 32 <= key <= 126 and len(self.input_name) < 25:
                self.input_name += chr(key)
        
        return False
    
    def run(self):
        """Main application loop."""
        print("\n" + "="*60)
        print("  TWO-STAGE LIVENESS VERIFICATION SYSTEM")
        print("="*60)
        print("  Stage 1: Blink your eyes (3 times)")
        print("  Stage 2: Follow head pose challenge")
        print("-"*60)
        print(f"  Registered faces: {self.storage.count()}")
        print("  SPACE=Start | R=Retry | Q=Quit")
        print("="*60 + "\n")
        
        cv2.namedWindow("Liveness Verification", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                display_frame = self.process_frame(frame)
                cv2.imshow("Liveness Verification", display_frame)
                
                if self.handle_key_input(cv2.waitKey(1)):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            print(f"\nTotal registered: {self.storage.count()}")


def main():
    """Entry point."""
    import sys
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    try:
        system = LivenessVerificationSystem(predictor_path)
        system.run()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
