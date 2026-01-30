# Team20_Byteme - Human Verification System
## Problem Statement 1: Human Verification through Blink Detection and Face Recognition

### Project Overview
A web-based human verification system that combines **real-time blink detection** with **face verification** to authenticate that a user is a real human. The system uses MediaPipe for facial landmark detection and face_recognition library for identity verification.

### Features
✅ **Live Video Feed** - Real-time webcam streaming with face detection  
✅ **Face Verification** - Automatic face registration and verification  
✅ **Blink Detection** - Detects eye blinks using facial landmarks  
✅ **Real-time Counter** - Shows blink count and face status  
✅ **Web Interface** - Beautiful responsive HTML/CSS UI  
✅ **Auto-verification** - Automatically verifies when both conditions are met (3 blinks + same face)  
✅ **Congratulations Screen** - Celebratory message with auto-close  

### Requirements
- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- Flask
- face_recognition
- NumPy

### Installation
```bash
pip install -r requirement.txt
```

### Running the Application
```bash
python app.py
```
Then open your browser to: **http://127.0.0.1:5000**

### How It Works
1. **Face Registration** - The system captures and registers your face for the first few seconds
2. **Real-time Monitoring** - MediaPipe tracks 468 facial landmarks including eye positions
3. **Blink Detection** - Calculates eye aspect ratio to detect blinks
4. **Face Verification** - Compares current face with registered face using face_recognition
5. **Success Criteria** - User must blink 3 times while maintaining the same face
6. **Verification Complete** - Shows congratulations screen and auto-closes

### File Structure
```
├── app.py                 # Flask application with video processing
├── blinkcounter.py        # Original blink counter logic
├── Facemeshmodule.py      # Face mesh detection wrapper
├── requirement.txt        # Python dependencies
├── templates/
│   └── index.html         # Web UI with real-time updates
```

### Technical Details

#### Blink Detection Algorithm
- Uses MediaPipe's 468-point facial mesh
- Tracks eye landmarks (indices 22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243)
- Calculates vertical to horizontal eye aspect ratio
- Blink detected when ratio < 33

#### Face Verification
- Extracts facial encodings using dlib's CNN model
- Compares encodings with tolerance of 0.6
- Face check runs every 10 frames for optimal performance

#### Performance Optimizations
- Face recognition runs every 10 frames (not every frame)
- Downscales image to 25% for face encoding
- Uses "hog" model for faster detection
- JPEG quality set to 80% for faster streaming
- Camera resolution: 640x480 at 30fps

### Requirements Met
✅ Real human verification through liveness detection  
✅ Face registration and verification  
✅ Blink counting (3 blinks required)  
✅ Smooth web interface with live updates  
✅ Auto-close on successful verification  

