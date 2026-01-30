# import install_requirements

import cv2
from Facemeshmodule import faceMeshDetection

# Use webcam (0) instead of video file since Video.mp4 doesn't exist
cap = cv2.VideoCapture(0)
detector = faceMeshDetection()


idList = [22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []

blinkCount = 0
counterTime = 0
color = (255,0,255)

run = True
while run:

    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from camera. Exiting...")
        break
        
    img, faces = detector.findFaceMesh(img,draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0,200,0), 3)
        cv2.line(img, leftLeft, leftRight, (0,200,0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = (sum(ratioList) / len(ratioList))

        if ratioAvg < 33 and counterTime == 0:
            blinkCount += 1
            counterTime = 1
            color = (0,255,0)
        if counterTime != 0:
            counterTime += 1
            if counterTime > 10:
                counterTime = 0
                color = (255,0,255)

        img = cv2.resize(img, (640, 360))
        print(f'Blink Count: {blinkCount}')
    
    cv2.imshow('Image',img)
    key = cv2.waitKey(25)
    if key == ord('q') or key == 27:  # Press 'q' or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()