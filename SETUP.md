# Setup Guide - Human Verification System

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Web Interface
Open your browser and go to: **http://127.0.0.1:5000**

## System Requirements
- Windows/Mac/Linux
- Python 3.8+
- Webcam
- 4GB RAM (recommended)

## Features & How to Use

### Face Registration
- Position your face clearly in the camera
- The system will automatically register your face (takes ~2 seconds)
- You'll see "📝 Registering..." status

### Blink Detection
- Once face is registered, naturally blink 3 times
- Each blink is counted and displayed in real-time
- Keep your face in the camera throughout the process

### Verification Success
- When you complete 3 blinks with the registered face:
  - Status badge turns green: "✓ HUMAN VERIFIED"
  - Congratulations overlay appears with animation
  - Application automatically closes after 2 seconds

## Troubleshooting

### Camera Not Working
- Check if your webcam is connected
- Verify no other application is using the webcam
- Try restarting the application

### Face Not Detected
- Ensure good lighting
- Keep your face centered in the video
- Remove sunglasses or large accessories

### Blinks Not Counting
- Blink more clearly
- Ensure your eyes are visible in the camera
- Keep the same face throughout the process

## Performance Tips
- Use good lighting for best face detection
- Keep camera at eye level
- Ensure stable internet connection if streaming

## Architecture

```
User Browser (HTML/CSS/JS)
       ↓
   Flask Server
       ↓
  Video Processing
   ├─ MediaPipe (Face Mesh)
   ├─ Face Recognition
   └─ Blink Detection
       ↓
  OpenCV (Camera Feed)
```

## API Endpoints

### `/` 
Main web interface

### `/video_feed`
Streams live video with face detection

### `/get_stats`
Returns current verification stats (JSON)
```json
{
  "blink_count": 2,
  "face_status": "verified",
  "verification_passed": false
}
```

### `/reset`
Resets the verification state

### `/shutdown`
Closes the application (auto-called on success)

## Configuration

To modify verification parameters, edit `app.py`:

```python
# Required blinks to pass
# Change line: if state.blink_count >= 3
state.blink_count >= 3  # Change 3 to desired number

# Face check interval (every N frames)
state.face_check_interval = 10  # Decrease for more frequent checks

# Face registration frames
state.required_registration_frames = 5  # Increase for longer registration
```

## Security Notes
- Face encodings are stored in memory only
- No data is persisted to disk
- Each session starts fresh
- Use HTTPS in production

## Support
For issues or questions, contact the development team.

---
**Last Updated:** January 30, 2026
