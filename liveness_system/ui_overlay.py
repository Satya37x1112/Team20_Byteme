"""
UI Overlay Module for Liveness Verification System
Handles all visual overlays including challenge display, progress indicators, and status
"""

import cv2
import numpy as np
from typing import Tuple
from . import config


class UIOverlay:
    """Creates professional, kiosk-like visual experience for liveness verification."""
    
    def __init__(self, frame_width: int, frame_height: int):
        """Initialize UI overlay with frame dimensions."""
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
        
        # Pre-create oval mask
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
        
        self.oval_mask_blur = cv2.GaussianBlur(self.oval_mask, (51, 51), 0)
        self.oval_mask_normalized = self.oval_mask_blur.astype(np.float32) / 255.0
    
    def apply_background_blur(self, frame: np.ndarray, blur_strength: int = 35) -> np.ndarray:
        """Apply blur to background while keeping face region clear."""
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        mask_3ch = np.stack([self.oval_mask_normalized] * 3, axis=-1)
        result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        return result
    
    def draw_oval_frame(self, frame: np.ndarray, status: str = 'default', thickness: int = 3) -> np.ndarray:
        """Draw oval frame overlay."""
        color_map = {
            'default': config.COLOR_OVAL,
            'active': config.COLOR_OVAL_ACTIVE,
            'success': config.COLOR_SUCCESS,
            'error': config.COLOR_ERROR,
            'warning': config.COLOR_WARNING,
            'challenge': config.COLOR_CHALLENGE
        }
        
        color = color_map.get(status, config.COLOR_OVAL)
        
        cv2.ellipse(
            frame,
            self.oval_center,
            self.oval_axes,
            0, 0, 360,
            color, thickness
        )
        
        self._draw_corner_markers(frame, color)
        return frame
    
    def _draw_corner_markers(self, frame: np.ndarray, color: Tuple[int, int, int]):
        """Draw corner markers around oval."""
        cx, cy = self.oval_center
        ax, ay = self.oval_axes
        marker_length = 20
        
        cv2.line(frame, (cx, cy - ay - 10), (cx, cy - ay - 10 - marker_length), color, 2)
        cv2.line(frame, (cx, cy + ay + 10), (cx, cy + ay + 10 + marker_length), color, 2)
        cv2.line(frame, (cx - ax - 10, cy), (cx - ax - 10 - marker_length, cy), color, 2)
        cv2.line(frame, (cx + ax + 10, cy), (cx + ax + 10 + marker_length, cy), color, 2)
    
    def draw_timer(self, frame: np.ndarray, seconds_remaining: int) -> np.ndarray:
        """Draw countdown timer on frame."""
        if seconds_remaining > 10:
            color = config.COLOR_INFO
        elif seconds_remaining > 5:
            color = config.COLOR_WARNING
        else:
            color = config.COLOR_ERROR
        
        timer_text = f"{seconds_remaining}s"
        text_size = cv2.getTextSize(timer_text, config.FONT, 1.5, 3)[0]
        
        x = self.frame_width - text_size[0] - 30
        y = 50
        
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            config.COLOR_TEXT_BG, -1
        )
        
        cv2.putText(frame, timer_text, (x, y), config.FONT, 1.5, color, 3)
        self._draw_timer_arc(frame, seconds_remaining, config.VERIFICATION_TIMEOUT)
        return frame
    
    def _draw_timer_arc(self, frame: np.ndarray, remaining: int, total: int):
        """Draw circular progress indicator."""
        center = (self.frame_width - 80, 80)
        radius = 35
        
        cv2.circle(frame, center, radius, (50, 50, 50), 2)
        angle = int(360 * remaining / total)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, config.COLOR_INFO, 3)
    
    def draw_challenge(self, frame: np.ndarray, challenge_text: str, progress: float = 0) -> np.ndarray:
        """Draw the current challenge prominently."""
        # Challenge background box
        box_height = 80
        box_y = 80
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, box_y), (self.frame_width - 20, box_y + box_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Challenge text
        text_size = cv2.getTextSize(challenge_text, config.FONT, 1.0, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        y = box_y + 35
        
        cv2.putText(frame, challenge_text, (x, y), config.FONT, 1.0, config.COLOR_CHALLENGE, 2)
        
        # Progress bar
        bar_width = self.frame_width - 80
        bar_height = 15
        bar_x = 40
        bar_y = box_y + 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress fill
        fill_width = int(bar_width * min(progress / 100, 1.0))
        if fill_width > 0:
            color = config.COLOR_SUCCESS if progress >= 100 else config.COLOR_INFO
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)
        
        # Progress text
        progress_text = f"{int(progress)}%"
        cv2.putText(frame, progress_text, (bar_x + bar_width + 10, bar_y + 12), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
    
    def draw_pose_indicator(self, frame: np.ndarray, yaw: float, pitch: float, 
                           challenge: str = None) -> np.ndarray:
        """Draw head pose indicator showing current orientation."""
        # Indicator position (bottom left)
        cx, cy = 80, self.frame_height - 80
        radius = 50
        
        # Draw background circle
        cv2.circle(frame, (cx, cy), radius, (50, 50, 50), -1)
        cv2.circle(frame, (cx, cy), radius, config.COLOR_INFO, 2)
        
        # Draw center dot
        cv2.circle(frame, (cx, cy), 5, config.COLOR_INFO, -1)
        
        # Draw current pose position (scaled and clamped)
        scale = 2
        dx = int(np.clip(yaw * scale, -radius + 10, radius - 10))
        dy = int(np.clip(pitch * scale, -radius + 10, radius - 10))
        
        pose_x = cx + dx
        pose_y = cy + dy
        
        cv2.circle(frame, (pose_x, pose_y), 10, config.COLOR_WARNING, -1)
        cv2.line(frame, (cx, cy), (pose_x, pose_y), config.COLOR_WARNING, 2)
        
        # Draw direction arrows based on challenge
        if challenge:
            arrow_color = config.COLOR_CHALLENGE
            arrow_length = 30
            
            if challenge == "LEFT":
                cv2.arrowedLine(frame, (cx - 10, cy), (cx - arrow_length, cy), 
                               arrow_color, 3, tipLength=0.4)
            elif challenge == "RIGHT":
                cv2.arrowedLine(frame, (cx + 10, cy), (cx + arrow_length, cy), 
                               arrow_color, 3, tipLength=0.4)
            elif challenge == "UP":
                cv2.arrowedLine(frame, (cx, cy - 10), (cx, cy - arrow_length), 
                               arrow_color, 3, tipLength=0.4)
            elif challenge == "DOWN":
                cv2.arrowedLine(frame, (cx, cy + 10), (cx, cy + arrow_length), 
                               arrow_color, 3, tipLength=0.4)
        
        # Labels
        cv2.putText(frame, f"Yaw: {yaw:.0f}", (cx - 35, cy + radius + 20), 
                   config.FONT, 0.4, config.COLOR_INFO, 1)
        cv2.putText(frame, f"Pitch: {pitch:.0f}", (cx - 35, cy + radius + 35), 
                   config.FONT, 0.4, config.COLOR_INFO, 1)
        
        return frame
    
    def draw_blink_counter(self, frame: np.ndarray, current_blinks: int, 
                          required_blinks: int, ear: float = None) -> np.ndarray:
        """Draw blink counter display during blink verification phase."""
        # Blink counter box at top
        box_height = 100
        box_y = 80
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, box_y), (self.frame_width - 20, box_y + box_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        title = "BLINK YOUR EYES"
        text_size = cv2.getTextSize(title, config.FONT, 0.9, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        cv2.putText(frame, title, (x, box_y + 30), config.FONT, 0.9, config.COLOR_BLINK, 2)
        
        # Blink counter with eye icons
        counter_y = box_y + 70
        icon_spacing = 60
        start_x = (self.frame_width - (required_blinks * icon_spacing)) // 2 + 20
        
        for i in range(required_blinks):
            icon_x = start_x + i * icon_spacing
            if i < current_blinks:
                # Filled eye (completed)
                self._draw_eye_icon(frame, icon_x, counter_y, True, config.COLOR_SUCCESS)
            else:
                # Empty eye (pending)
                self._draw_eye_icon(frame, icon_x, counter_y, False, (100, 100, 100))
        
        # Progress text
        progress_text = f"{current_blinks}/{required_blinks} blinks"
        text_size = cv2.getTextSize(progress_text, config.FONT, 0.6, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        cv2.putText(frame, progress_text, (x, box_y + box_height + 25), 
                   config.FONT, 0.6, config.COLOR_INFO, 2)
        
        # EAR indicator (bottom left)
        if ear is not None:
            self._draw_ear_indicator(frame, ear)
        
        return frame
    
    def _draw_eye_icon(self, frame: np.ndarray, cx: int, cy: int, 
                       filled: bool, color: Tuple[int, int, int]):
        """Draw a simple eye icon."""
        # Eye shape (ellipse)
        axes = (20, 12)
        thickness = 2 if not filled else -1
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, color, thickness)
        
        # Pupil
        if filled:
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 0), -1)
    
    def _draw_ear_indicator(self, frame: np.ndarray, ear: float):
        """Draw Eye Aspect Ratio indicator."""
        # Position (bottom left)
        bar_x = 20
        bar_y = self.frame_height - 100
        bar_width = 100
        bar_height = 20
        
        # Label
        cv2.putText(frame, "EAR", (bar_x, bar_y - 5), config.FONT, 0.4, config.COLOR_INFO, 1)
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # EAR fill (normalized: 0.15-0.35 typical range)
        normalized_ear = np.clip((ear - 0.15) / 0.20, 0, 1)
        fill_width = int(bar_width * normalized_ear)
        
        # Color based on threshold
        if ear < config.EAR_THRESHOLD:
            color = config.COLOR_BLINK  # Eyes closed
        else:
            color = config.COLOR_SUCCESS  # Eyes open
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Threshold line
        threshold_x = bar_x + int(bar_width * (config.EAR_THRESHOLD - 0.15) / 0.20)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                config.COLOR_WARNING, 2)
        
        # Value text
        cv2.putText(frame, f"{ear:.2f}", (bar_x + bar_width + 10, bar_y + 15), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
    
    def draw_phase_indicator(self, frame: np.ndarray, current_phase: int, 
                            phase_names: list = None) -> np.ndarray:
        """Draw phase progress indicator (Phase 1: Blink, Phase 2: Head Pose)."""
        if phase_names is None:
            phase_names = ["Blink Check", "Head Pose"]
        
        # Position at top left
        start_x = 20
        start_y = 30
        
        for i, name in enumerate(phase_names):
            x = start_x + i * 120
            
            # Circle
            if i < current_phase:
                # Completed
                cv2.circle(frame, (x, start_y), 12, config.COLOR_SUCCESS, -1)
                cv2.putText(frame, "✓", (x - 6, start_y + 5), config.FONT, 0.5, (255, 255, 255), 2)
            elif i == current_phase:
                # Current
                cv2.circle(frame, (x, start_y), 12, config.COLOR_BLINK if i == 0 else config.COLOR_CHALLENGE, 2)
                cv2.putText(frame, str(i + 1), (x - 5, start_y + 5), config.FONT, 0.5, 
                           config.COLOR_BLINK if i == 0 else config.COLOR_CHALLENGE, 2)
            else:
                # Pending
                cv2.circle(frame, (x, start_y), 12, (100, 100, 100), 2)
                cv2.putText(frame, str(i + 1), (x - 5, start_y + 5), config.FONT, 0.5, (100, 100, 100), 1)
            
            # Label
            cv2.putText(frame, name, (x - 30, start_y + 30), config.FONT, 0.4, 
                       config.COLOR_INFO if i <= current_phase else (100, 100, 100), 1)
            
            # Connector line
            if i < len(phase_names) - 1:
                line_start = x + 15
                line_end = x + 105
                color = config.COLOR_SUCCESS if i < current_phase else (100, 100, 100)
                cv2.line(frame, (line_start, start_y), (line_end, start_y), color, 2)
        
        return frame

    def draw_message(self, frame: np.ndarray, message: str, status: str = 'info', 
                     position: str = 'bottom') -> np.ndarray:
        """Draw status message on frame."""
        color_map = {
            'info': config.COLOR_INFO,
            'success': config.COLOR_SUCCESS,
            'warning': config.COLOR_WARNING,
            'error': config.COLOR_ERROR
        }
        color = color_map.get(status, config.COLOR_INFO)
        
        text_size = cv2.getTextSize(message, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
        x = (self.frame_width - text_size[0]) // 2
        
        if position == 'top':
            y = 40
        elif position == 'center':
            y = self.frame_height // 2
        else:
            y = self.frame_height - 40
        
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            config.COLOR_TEXT_BG, -1
        )
        
        cv2.putText(frame, message, (x, y), config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS)
        return frame
    
    def draw_face_box(self, frame: np.ndarray, face_location: Tuple[int, int, int, int],
                      status: str = 'default') -> np.ndarray:
        """Draw rectangle around detected face."""
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
        """Draw success screen after verification."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        center = (self.frame_width // 2, self.frame_height // 2 - 50)
        cv2.circle(frame, center, 60, config.COLOR_SUCCESS, 3)
        
        pts = np.array([
            [center[0] - 30, center[1]],
            [center[0] - 10, center[1] + 25],
            [center[0] + 35, center[1] - 25]
        ], np.int32)
        cv2.polylines(frame, [pts], False, config.COLOR_SUCCESS, 4)
        
        msg1 = "Liveness Verified!"
        msg2 = f"Registered: {name}"
        
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
        """Draw failure screen."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        center = (self.frame_width // 2, self.frame_height // 2 - 50)
        cv2.circle(frame, center, 60, config.COLOR_ERROR, 3)
        cv2.line(frame, (center[0] - 25, center[1] - 25), 
                (center[0] + 25, center[1] + 25), config.COLOR_ERROR, 4)
        cv2.line(frame, (center[0] + 25, center[1] - 25), 
                (center[0] - 25, center[1] + 25), config.COLOR_ERROR, 4)
        
        msg1 = "Liveness Verification Failed"
        
        text_size1 = cv2.getTextSize(msg1, config.FONT, 0.9, 2)[0]
        text_size2 = cv2.getTextSize(reason, config.FONT, 0.6, 2)[0]
        
        x1 = (self.frame_width - text_size1[0]) // 2
        x2 = (self.frame_width - text_size2[0]) // 2
        
        cv2.putText(frame, msg1, (x1, self.frame_height // 2 + 50), 
                   config.FONT, 0.9, config.COLOR_ERROR, 2)
        cv2.putText(frame, reason, (x2, self.frame_height // 2 + 90), 
                   config.FONT, 0.6, config.COLOR_WARNING, 2)
        
        retry_msg = "Press 'R' to retry or 'Q' to quit"
        text_size3 = cv2.getTextSize(retry_msg, config.FONT, 0.5, 1)[0]
        x3 = (self.frame_width - text_size3[0]) // 2
        cv2.putText(frame, retry_msg, (x3, self.frame_height // 2 + 130), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
    
    def draw_recognized_screen(self, frame: np.ndarray, name: str) -> np.ndarray:
        """Draw screen when a known face is recognized."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 80, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        center = (self.frame_width // 2, self.frame_height // 2 - 60)
        cv2.circle(frame, center, 70, config.COLOR_SUCCESS, 4)
        
        head_center = (center[0], center[1] - 15)
        cv2.circle(frame, head_center, 20, config.COLOR_SUCCESS, 3)
        cv2.ellipse(frame, (center[0], center[1] + 30), (30, 25), 0, 180, 360, config.COLOR_SUCCESS, 3)
        
        welcome = "Welcome Back!"
        
        text_size1 = cv2.getTextSize(welcome, config.FONT, 1.0, 2)[0]
        text_size2 = cv2.getTextSize(name, config.FONT, 1.2, 3)[0]
        
        x1 = (self.frame_width - text_size1[0]) // 2
        x2 = (self.frame_width - text_size2[0]) // 2
        
        cv2.putText(frame, welcome, (x1, self.frame_height // 2 + 50), 
                   config.FONT, 1.0, config.COLOR_SUCCESS, 2)
        cv2.putText(frame, name, (x2, self.frame_height // 2 + 100), 
                   config.FONT, 1.2, (255, 255, 255), 3)
        
        hint = "Press SPACE to continue"
        text_size3 = cv2.getTextSize(hint, config.FONT, 0.5, 1)[0]
        cv2.putText(frame, hint, ((self.frame_width - text_size3[0]) // 2, self.frame_height - 40), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
    
    def draw_name_input_screen(self, frame: np.ndarray, current_name: str) -> np.ndarray:
        """Draw name input screen."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        center = (self.frame_width // 2, self.frame_height // 2 - 100)
        cv2.circle(frame, center, 40, config.COLOR_SUCCESS, 2)
        pts = np.array([
            [center[0] - 20, center[1]],
            [center[0] - 5, center[1] + 15],
            [center[0] + 25, center[1] - 15]
        ], np.int32)
        cv2.polylines(frame, [pts], False, config.COLOR_SUCCESS, 3)
        
        title = "Challenge Completed!"
        text_size = cv2.getTextSize(title, config.FONT, 0.8, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        cv2.putText(frame, title, (x, self.frame_height // 2 - 30), 
                   config.FONT, 0.8, config.COLOR_SUCCESS, 2)
        
        prompt = "Enter your name:"
        text_size = cv2.getTextSize(prompt, config.FONT, 0.7, 2)[0]
        x = (self.frame_width - text_size[0]) // 2
        cv2.putText(frame, prompt, (x, self.frame_height // 2 + 20), 
                   config.FONT, 0.7, config.COLOR_INFO, 2)
        
        box_width = 300
        box_height = 50
        box_x = (self.frame_width - box_width) // 2
        box_y = self.frame_height // 2 + 40
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     config.COLOR_INFO, 2)
        cv2.rectangle(frame, (box_x + 2, box_y + 2), (box_x + box_width - 2, box_y + box_height - 2), 
                     (30, 30, 30), -1)
        
        display_name = current_name + "|"
        cv2.putText(frame, display_name, (box_x + 10, box_y + 35), 
                   config.FONT, 0.8, (255, 255, 255), 2)
        
        hint1 = "Type your name and press ENTER"
        hint2 = "Press ESC to cancel"
        
        text_size1 = cv2.getTextSize(hint1, config.FONT, 0.5, 1)[0]
        text_size2 = cv2.getTextSize(hint2, config.FONT, 0.5, 1)[0]
        
        cv2.putText(frame, hint1, ((self.frame_width - text_size1[0]) // 2, box_y + box_height + 30), 
                   config.FONT, 0.5, config.COLOR_INFO, 1)
        cv2.putText(frame, hint2, ((self.frame_width - text_size2[0]) // 2, box_y + box_height + 55), 
                   config.FONT, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw initial instructions overlay."""
        instructions = [
            "1. Position your face inside the oval",
            "2. Ensure good lighting",
            "3. Follow the head movement challenge",
            "Press SPACE to start verification"
        ]
        
        y_start = self.frame_height - 120
        for i, instruction in enumerate(instructions):
            y = y_start + i * 25
            cv2.putText(frame, instruction, (20, y), config.FONT, 0.5, config.COLOR_INFO, 1)
        
        return frame
