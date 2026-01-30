from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from Facemeshmodule import faceMeshDetection
import face_recognition
import threading

app = Flask(__name__)

# Global variables for state management
class VerificationState:
    def __init__(self):
        self.blink_count = 0
        self.counter_time = 0
        self.ratio_list = []
        self.face_status = "detecting"
        self.verification_passed = False
        self.registered_face_encoding = None
        self.face_registration_frames = 0
        self.required_registration_frames = 5  # Reduced from 10
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_face_encoding = None
        self.face_check_interval = 10  # Check face every 10 frames instead of every frame
        
    def reset(self):
        with self.lock:
            self.blink_count = 0
            self.counter_time = 0
            self.ratio_list = []
            self.face_status = "detecting"
            self.verification_passed = False
            self.registered_face_encoding = None
            self.face_registration_frames = 0
            self.frame_count = 0
            self.last_face_encoding = None

state = VerificationState()

# Initialize face mesh detector
detector = faceMeshDetection()

# Camera capture
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

# Landmark indices for eye detection
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

def generate_frames():
    global state
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        state.frame_count += 1
        
        # Process frame for face mesh (this is fast)
        img, faces = detector.findFaceMesh(frame, draw=False)
        
        # Only do face recognition every N frames to reduce lag
        do_face_recognition = (state.frame_count % state.face_check_interval == 0)
        
        with state.lock:
            if faces:
                face = faces[0]
                
                # Face recognition only every N frames
                if do_face_recognition:
                    # Use smaller image for face recognition (faster)
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small, model="hog")
                    
                    if len(face_locations) > 0:
                        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                        
                        if len(face_encodings) > 0:
                            current_encoding = face_encodings[0]
                            state.last_face_encoding = current_encoding
                            
                            # Face registration phase
                            if state.registered_face_encoding is None:
                                state.face_registration_frames += 1
                                state.face_status = "registering"
                                
                                if state.face_registration_frames >= state.required_registration_frames:
                                    state.registered_face_encoding = current_encoding
                                    state.face_status = "verified"
                            else:
                                # Face verification
                                matches = face_recognition.compare_faces(
                                    [state.registered_face_encoding], 
                                    current_encoding,
                                    tolerance=0.6
                                )
                                
                                if matches[0]:
                                    state.face_status = "verified"
                                else:
                                    state.face_status = "mismatch"
                                    state.verification_passed = False
                    else:
                        if state.registered_face_encoding is None:
                            state.face_status = "detecting"
                
                # Draw landmarks for eyes (always do this - it's fast)
                color = (255, 0, 255) if state.counter_time == 0 else (0, 255, 0)
                for id in idList:
                    if id < len(face):
                        cv2.circle(img, tuple(face[id]), 3, color, cv2.FILLED)
                
                # Calculate blink ratio (always do this - it's fast)
                if len(face) > max(idList):
                    leftUp = face[159]
                    leftDown = face[23]
                    leftLeft = face[130]
                    leftRight = face[243]
                    
                    lengthVer, _ = detector.findDistance(leftUp, leftDown)
                    lengthHor, _ = detector.findDistance(leftLeft, leftRight)
                    
                    cv2.line(img, tuple(leftUp), tuple(leftDown), (0, 200, 0), 2)
                    cv2.line(img, tuple(leftLeft), tuple(leftRight), (0, 200, 0), 2)
                    
                    if lengthHor > 0:
                        ratio = int((lengthVer / lengthHor) * 100)
                        state.ratio_list.append(ratio)
                        if len(state.ratio_list) > 3:
                            state.ratio_list.pop(0)
                        ratioAvg = sum(state.ratio_list) / len(state.ratio_list)
                        
                        # Blink detection
                        if ratioAvg < 33 and state.counter_time == 0 and state.face_status == "verified":
                            state.blink_count += 1
                            state.counter_time = 1
                        
                        if state.counter_time != 0:
                            state.counter_time += 1
                            if state.counter_time > 10:
                                state.counter_time = 0
                
                # Draw face box based on face mesh bounds
                if len(face) > 0:
                    x_coords = face[:, 0]
                    y_coords = face[:, 1]
                    left, right = int(np.min(x_coords)), int(np.max(x_coords))
                    top, bottom = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    box_color = (0, 255, 0) if state.face_status == "verified" else (0, 255, 255) if state.face_status == "registering" else (0, 0, 255)
                    cv2.rectangle(img, (left - 10, top - 10), (right + 10, bottom + 10), box_color, 2)
                    
                    # Status text on video
                    status_text = f"Face: {state.face_status.upper()}"
                    cv2.putText(img, status_text, (left - 10, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Check verification success
                if state.blink_count >= 3 and state.face_status == "verified":
                    state.verification_passed = True
                    
            else:
                if state.registered_face_encoding is None:
                    state.face_status = "detecting"
                else:
                    state.face_status = "no_face"
        
        # Add overlay information
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        
        cv2.putText(img, f"Blinks: {state.blink_count}/3", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if state.verification_passed else (255, 255, 255)
        status_text = "VERIFIED!" if state.verification_passed else "In Progress..."
        cv2.putText(img, status_text, (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Encode frame with lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    with state.lock:
        return jsonify({
            'blink_count': state.blink_count,
            'face_status': state.face_status,
            'verification_passed': state.verification_passed
        })

@app.route('/reset')
def reset():
    state.reset()
    return jsonify({'status': 'success'})

@app.route('/shutdown')
def shutdown():
    import os
    print("\n" + "="*50)
    print("✅ VERIFICATION SUCCESSFUL!")
    print("🎉 Human verified - Closing application...")
    print("="*50 + "\n")
    # Release camera
    camera.release()
    # Exit the application
    os._exit(0)
    return jsonify({'status': 'shutting down'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔐 Human Verification System Started!")
    print("="*50)
    print("\n📍 Open your browser and go to: http://127.0.0.1:5000")
    print("\n📋 Instructions:")
    print("   1. Position your face in the camera")
    print("   2. Wait for face registration")
    print("   3. Blink 5 times naturally")
    print("   4. Keep the same face throughout")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
