"""
UI Overlay Module
Handles all visual overlays including blur, oval frame, text, and status indicators
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from . import config


class UIOverlay:
    """
    Manages UI overlays for the registration interface.
    Creates professional, kiosk-like visual experience.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize UI overlay with frame dimensions.
        
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
        
        # Pre-create oval mask for efficiency
        self._create_oval_mask()
    
    def _create_oval_mask(self):
        """Create oval mask for background blur effect."""
        self.oval_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.ellipse(
            self.oval_mask,
            self.oval_center,
            self.oval_axes,
            0, 0, 360,
            255, -1
        )
        
        # Create feathered edge for smooth transition
        self.oval_mask_blur = cv2.GaussianBlur(self.oval_mask, (51, 51), 0)
        self.oval_mask_normalized = self.oval_mask_blur.astype(np.float32) / 255.0
    
    def apply_background_blur(self, frame: np.ndarray, blur_strength: int = 35) -> np.ndarray:
        """
        Apply blur to background while keeping face region clear.
        
        Args:
            frame: Original BGR frame
            blur_strength: Gaussian blur kernel size (odd number)
            
        Returns:
            Frame with blurred background
        """
        # Create heavily blurred version
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Blend original and blurred using mask
        mask_3ch = np.stack([self.oval_mask_normalized] * 3, axis=-1)
        result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def draw_oval_frame(self, frame: np.ndarray, status: str = 'default', thickness: int = 3) -> np.ndarray:
        """
        Draw oval frame overlay.
        
        Args:
            frame: BGR frame to draw on
            status: 'default', 'active', 'success', 'error'
            thickness: Line thickness
            
        Returns:
            Frame with oval overlay
        """
        color_map = {
            'default': config.COLOR_OVAL,
            'active': config.COLOR_OVAL_ACTIVE,
            'success': config.COLOR_SUCCESS,
            'error': config.COLOR_ERROR,
            'warning': config.COLOR_WARNING
        }
        
        color = color_map.get(status, config.COLOR_OVAL)
        
        # Draw main oval
        cv2.ellipse(
            frame,
            self.oval_center,
            self.oval_axes,
            0, 0, 360,
            color, thickness
        )
        
        # Draw corner markers for professional look
        self._draw_corner_markers(frame, color)
        
        return frame
    
    def _draw_corner_markers(self, frame: np.ndarray, color: Tuple[int, int, int]):
        """Draw corner markers around oval for visual guidance."""
        cx, cy = self.oval_center
        ax, ay = self.oval_axes
        marker_length = 20
        
        # Top marker
        cv2.line(frame, (cx, cy - ay - 10), (cx, cy - ay - 10 - marker_length), color, 2)
        
        # Bottom marker
        cv2.line(frame, (cx, cy + ay + 10), (cx, cy + ay + 10 + marker_length), color, 2)
        
        # Left marker
        cv2.line(frame, (cx - ax - 10, cy), (cx - ax - 10 - marker_length, cy), color, 2)
        
        # Right marker
        cv2.line(frame, (cx + ax + 10, cy), (cx + ax + 10 + marker_length, cy), color, 2)
    
    def draw_timer(self, frame: np.ndarray, seconds_remaining: int) -> np.ndarray:
        """
        Draw countdown timer on frame.
        
        Args:
            frame: BGR frame
            seconds_remaining: Seconds left
            
        Returns:
            Frame with timer overlay
        """
        # Determine color based on time remaining
        if seconds_remaining > 10:
            color = config.COLOR_INFO
        elif seconds_remaining > 5:
            color = config.COLOR_WARNING
        else:
            color = config.COLOR_ERROR
        
        # Draw timer background
        timer_text = f"{seconds_remaining}s"
        text_size = cv2.getTextSize(timer_text, config.FONT, 1.5, 3)[0]
        
        # Position at top right
        x = self.frame_width - text_size[0] - 30
        y = 50
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            config.COLOR_TEXT_BG,
            -1
        )
        
        # Draw timer text
        cv2.putText(frame, timer_text, (x, y), config.FONT, 1.5, color, 3)
        
        # Draw progress arc
        self._draw_timer_arc(frame, seconds_remaining, config.REGISTRATION_TIMEOUT)
        
        return frame
    
    def _draw_timer_arc(self, frame: np.ndarray, remaining: int, total: int):
        """Draw circular progress indicator for timer."""
        center = (self.frame_width - 80, 80)
        radius = 35
        
        # Background circle
        cv2.circle(frame, center, radius, (50, 50, 50), 2)
        
        # Progress arc
        angle = int(360 * remaining / total)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, config.COLOR_INFO, 3)
    
    def draw_message(self, frame: np.ndarray, message: str, status: str = 'info', 
                     position: str = 'bottom') -> np.ndarray:
        """
        Draw status message on frame.
        
        Args:
            frame: BGR frame
            message: Message text
            status: 'info', 'success', 'warning', 'error'
            position: 'top', 'bottom', 'center'
            
        Returns:
            Frame with message overlay
        """
        color_map = {
            'info': config.COLOR_INFO,
            'success': config.COLOR_SUCCESS,
            'warning': config.COLOR_WARNING,
            'error': config.COLOR_ERROR
        }
        color = color_map.get(status, config.COLOR_INFO)
        
        # Calculate text size
        text_size = cv2.getTextSize(message, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
        
        # Position calculation
        x = (self.frame_width - text_size[0]) // 2
        
        if position == 'top':
            y = 40
        elif position == 'center':
            y = self.frame_height // 2
        else:  # bottom
            y = self.frame_height - 40
        
        # Draw background
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            config.COLOR_TEXT_BG,
            -1
        )
        
        # Draw text
        cv2.putText(frame, message, (x, y), config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS)
        
        return frame
    
    def draw_blink_indicator(self, frame: np.ndarray, blink_count: int, 
                             required_blinks: int, eyes_closed: bool = False) -> np.ndarray:
        """
        Draw blink counter and eye status indicator.
        
        Args:
            frame: BGR frame
            blink_count: Current blink count
            required_blinks: Required blinks for verification
            eyes_closed: Whether eyes are currently closed
            
        Returns:
            Frame with blink indicator
        """
        # Position at top left
        x, y = 20, 40
        
        # Draw blink counter
        blink_text = f"Blinks: {blink_count}/{required_blinks}"
        color = config.COLOR_SUCCESS if blink_count >= required_blinks else config.COLOR_INFO
        
        cv2.putText(frame, blink_text, (x, y), config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS)
        
        # Draw eye status indicator
        eye_status = "Eyes: CLOSED" if eyes_closed else "Eyes: OPEN"
        eye_color = config.COLOR_WARNING if eyes_closed else config.COLOR_SUCCESS
        cv2.putText(frame, eye_status, (x, y + 30), config.FONT, 0.5, eye_color, 1)
        
        return frame
    
    def draw_face_box(self, frame: np.ndarray, face_location: Tuple[int, int, int, int],
                      status: str = 'default') -> np.ndarray:
        """
        Draw rectangle around detected face.
        
        Args:
            frame: BGR frame
            face_location: (top, right, bottom, left) tuple
            status: 'default', 'success', 'error'
            
        Returns:
            Frame with face box
        """
        color_map = {
            'default': config.COLOR_INFO,
            'success': config.COLOR_SUCCESS,
            'error': config.COLOR_ERROR
        }
        color = color_map.get(status, config.COLOR_INFO)
        
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        return frame
    
    def draw_success_screen(self, frame: np.ndarray, name: str) -> np.ndarray:
        """
        Draw success screen after registration.
        
        Args:
            frame: BGR frame
            name: Registered person's name
            
        Returns:
            Frame with success overlay
        """
        # Darken background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw success checkmark
        center = (self.frame_width // 2, self.frame_height // 2 - 50)
        cv2.circle(frame, center, 60, config.COLOR_SUCCESS, 3)
        
        # Draw checkmark
        pts = np.array([
            [center[0] - 30, center[1]],
            [center[0] - 10, center[1] + 25],
            [center[0] + 35, center[1] - 25]
        ], np.int32)
        cv2.polylines(frame, [pts], False, config.COLOR_SUCCESS, 4)
        
        # Draw success message
        msg1 = "Face Registered Successfully!"
        msg2 = f"Welcome, {name}"
        
        text_size1 = cv2.getTextSize(msg1, config.FONT, 0.9, 2)[0]
        text_size2 = cv2.getTextSize(msg2, config.FONT, 0.7, 2)[0]
        
        x1 = (self.frame_width - text_size1[0]) // 2
        x2 = (self.frame_width - text_size2[0]) // 2
        
        cv2.putText(frame, msg1, (x1, self.frame_height // 2 + 50), 
                   config.FONT, 0.9, config.COLOR_SUCCESS, 2)
        cv2.putText(frame, msg2, (x2, self.frame_height // 2 + 90), 
                   config.FONT, 0.7, config.COLOR_INFO, 2)
        
        return frame
    
    def draw_failure_screen(self, frame: np.ndarray, reason: str) -> np.ndarray:
        """
        Draw failure screen.
        
        Args:
            frame: BGR frame
            reason: Failure reason
            
        Returns:
            Frame with failure overlay
        """
        # Darken background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw X mark
        center = (self.frame_width // 2, self.frame_height // 2 - 50)
        cv2.circle(frame, center, 60, config.COLOR_ERROR, 3)
        cv2.line(frame, (center[0] - 25, center[1] - 25), 
                (center[0] + 25, center[1] + 25), config.COLOR_ERROR, 4)
        cv2.line(frame, (center[0] + 25, center[1] - 25), 
                (center[0] - 25, center[1] + 25), config.COLOR_ERROR, 4)
        
        # Draw failure message
        msg1 = "Registration Failed"
        
        text_size1 = cv2.getTextSize(msg1, config.FONT, 0.9, 2)[0]
        text_size2 = cv2.getTextSize(reason, config.FONT, 0.6, 2)[0]
        
        x1 = (self.frame_width - text_size1[0]) // 2
        x2 = (self.frame_width - text_size2[0]) // 2
        
        cv2.putText(frame, msg1, (x1, self.frame_height // 2 + 50), 
                   config.FONT, 0.9, config.COLOR_ERROR, 2)
        cv2.putText(frame, reason, (x2, self.frame_height // 2 + 90), 
                   config.FONT, 0.6, config.COLOR_WARNING, 2)
        
        # Draw retry instruction
        retry_msg = "Press 'R' to retry or 'Q' to quit"
        text_size3 = cv2.getTextSize(retry_msg, config.FONT, 0.5, 1)[0]
        x3 = (self.frame_width - text_size3[0]) // 2
        cv2.putText(frame, retry_msg, (x3, self.frame_height // 2 + 130), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw initial instructions overlay.
        
        Args:
            frame: BGR frame
            
        Returns:
            Frame with instructions
        """
        instructions = [
            "1. Position your face inside the oval",
            "2. Ensure good lighting",
            "3. Blink naturally when prompted",
            "Press SPACE to start registration"
        ]
        
        y_start = self.frame_height - 120
        for i, instruction in enumerate(instructions):
            y = y_start + i * 25
            cv2.putText(frame, instruction, (20, y), config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
