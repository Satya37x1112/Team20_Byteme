"""
Random Challenge Generator Module
Generates unpredictable liveness challenges to prevent replay attacks
"""

import random
import time
from typing import Tuple
from . import config


class ChallengeGenerator:
    """
    Generates random liveness challenges.
    Ensures unpredictability to defeat pre-recorded video attacks.
    """
    
    def __init__(self):
        """Initialize the challenge generator."""
        self.current_challenge = None
        self.challenge_issued_at = None
        self.thresholds = None
        self.challenge_history = []  # Track recent challenges
        
    def generate_challenge(self) -> Tuple[str, str, dict]:
        """
        Generate a new random challenge.
        
        Returns:
            Tuple of (challenge_code, display_text, thresholds)
        """
        # Get fresh random thresholds each time
        self.thresholds = config.get_pose_thresholds()
        
        # Avoid repeating the last 2 challenges if possible
        available = config.CHALLENGES.copy()
        if len(self.challenge_history) >= 1:
            last = self.challenge_history[-1]
            if last in available and len(available) > 1:
                available.remove(last)
        
        # Random selection
        self.current_challenge = random.choice(available)
        self.challenge_issued_at = time.time()
        
        # Track history
        self.challenge_history.append(self.current_challenge)
        if len(self.challenge_history) > 5:
            self.challenge_history.pop(0)
        
        display_text = config.CHALLENGE_DISPLAY_NAMES[self.current_challenge]
        
        return self.current_challenge, display_text, self.thresholds
    
    def get_current_challenge(self) -> Tuple[str, str]:
        """Get the current active challenge."""
        if self.current_challenge is None:
            return None, None
        
        display_text = config.CHALLENGE_DISPLAY_NAMES[self.current_challenge]
        return self.current_challenge, display_text
    
    def get_threshold(self, direction: str) -> float:
        """Get the threshold for a specific direction."""
        if self.thresholds is None:
            self.thresholds = config.get_pose_thresholds()
        return self.thresholds.get(direction, 15)
    
    def check_challenge_completed(self, yaw: float, pitch: float) -> bool:
        """
        Check if the current challenge is completed based on head pose.
        
        Args:
            yaw: Head yaw angle in degrees (left/right)
            pitch: Head pitch angle in degrees (up/down)
            
        Returns:
            True if challenge is completed
        """
        if self.current_challenge is None or self.thresholds is None:
            return False
        
        challenge = self.current_challenge
        
        if challenge == "LEFT":
            # Head turned left = positive yaw (from mirrored camera view)
            return yaw > self.thresholds["LEFT"]
        elif challenge == "RIGHT":
            # Head turned right = negative yaw (from mirrored camera view)
            return yaw < self.thresholds["RIGHT"]
        elif challenge == "UP":
            return pitch < self.thresholds["UP"]
        elif challenge == "DOWN":
            return pitch > self.thresholds["DOWN"]
        
        return False
    
    def get_progress_feedback(self, yaw: float, pitch: float) -> Tuple[float, str]:
        """
        Get progress feedback for the current challenge.
        
        Args:
            yaw: Current yaw angle
            pitch: Current pitch angle
            
        Returns:
            Tuple of (progress_percentage, feedback_message)
        """
        if self.current_challenge is None:
            return 0, ""
        
        challenge = self.current_challenge
        threshold = self.get_threshold(challenge)
        
        if challenge == "LEFT":
            # Positive yaw = head turned left (mirrored view)
            progress = min(100, (yaw / threshold) * 100) if yaw > 0 else 0
            if yaw < -5:
                return 0, "Turn LEFT, not right"
        elif challenge == "RIGHT":
            # Negative yaw = head turned right (mirrored view)
            progress = min(100, (abs(yaw) / abs(threshold)) * 100) if yaw < 0 else 0
            if yaw > 5:
                return 0, "Turn RIGHT, not left"
        elif challenge == "UP":
            progress = min(100, (abs(pitch) / abs(threshold)) * 100) if pitch < 0 else 0
            if pitch > 5:
                return 0, "Look UP, not down"
        elif challenge == "DOWN":
            progress = min(100, (pitch / threshold) * 100) if pitch > 0 else 0
            if pitch < -5:
                return 0, "Look DOWN, not up"
        else:
            progress = 0
        
        return progress, ""
    
    def reset(self):
        """Reset the challenge generator."""
        self.current_challenge = None
        self.challenge_issued_at = None
        self.thresholds = None
