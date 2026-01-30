import cv2
import math
import numpy as np
import urllib.request
import os

class faceMeshDetection:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Download face landmarker model if not exists
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        # Use OpenCV's face detection with Haar cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Download dlib's facial landmark predictor
        self.landmark_model_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
        
        # Try to use mediapipe if available
        self.use_mediapipe = False
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp
            
            if not os.path.exists(model_path):
                print("[LOG] Downloading face landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[LOG] Model downloaded successfully!")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=maxFaces,
                min_face_detection_confidence=minDetectionCon,
                min_tracking_confidence=minTrackCon
            )
            self.faceLandmarker = vision.FaceLandmarker.create_from_options(options)
            self.mp = mp
            self.use_mediapipe = True
            print("[LOG] Using MediaPipe for face detection")
        except Exception as e:
            print(f"[LOG] MediaPipe not available ({e}), using OpenCV Haar cascade")
            self.use_mediapipe = False

    def findFaceMesh(self, img, draw=True):
        faces = []
        
        if self.use_mediapipe:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=imgRGB)
            results = self.faceLandmarker.detect(mp_image)
            
            if results.face_landmarks:
                for faceLms in results.face_landmarks:
                    if draw:
                        for lm in faceLms:
                            x = int(lm.x * img.shape[1])
                            y = int(lm.y * img.shape[0])
                            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                    
                    face = np.array([[int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] for lm in faceLms])
                    faces.append(face)
        else:
            # Use OpenCV Haar cascade as fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in detected_faces[:self.maxFaces]:
                if draw:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Create approximate landmark points for eye regions
                # These are approximate positions based on face bounding box
                face_landmarks = []
                
                # Generate 478 landmark points (same as mediapipe) - approximate
                for i in range(478):
                    # Map landmark indices to approximate face positions
                    lx = x + int((i % 30) * w / 30)
                    ly = y + int((i // 30) * h / 16)
                    face_landmarks.append([lx, ly])
                
                # Key eye landmarks (approximate positions)
                # Left eye landmarks
                eye_y = y + int(h * 0.35)
                left_eye_x = x + int(w * 0.3)
                right_eye_x = x + int(w * 0.7)
                
                # Update key indices used in blinkcounter.py
                face_landmarks[22] = [left_eye_x, eye_y - 5]
                face_landmarks[23] = [left_eye_x, eye_y + 10]
                face_landmarks[24] = [left_eye_x - 5, eye_y]
                face_landmarks[26] = [left_eye_x + 5, eye_y]
                face_landmarks[110] = [left_eye_x, eye_y]
                face_landmarks[130] = [left_eye_x - 15, eye_y]
                face_landmarks[157] = [left_eye_x, eye_y - 8]
                face_landmarks[158] = [left_eye_x, eye_y - 6]
                face_landmarks[159] = [left_eye_x, eye_y - 10]  # leftUp
                face_landmarks[160] = [left_eye_x, eye_y - 4]
                face_landmarks[161] = [left_eye_x, eye_y - 2]
                face_landmarks[243] = [left_eye_x + 15, eye_y]  # leftRight
                
                faces.append(np.array(face_landmarks))
        
        return img, faces


    @staticmethod
    def findDistance(p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = np.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()