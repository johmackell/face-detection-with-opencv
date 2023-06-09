"""
Face detection using OpenCV for my science assessment.


Specific controls are:
esc - exit the program
1-9 - change scale factor
"""

import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCasc = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("----------\nWEBCAM NOT FOUND")
    sys.exit()

scale_factor = 1.1
detection_level = 1

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCasc.detectMultiScale(
        gray,
        scaleFactor = scale_factor,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for fX, fY, fW, fH in faces:
        cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF

    d = {
        49: 1.1,
        50: 1.4,
        51: 1.7,
        52: 2.0,
        53: 2.3,
        54: 2.6,
        55: 2.9,
        56: 3.2,
        57: 3.5
    }

    if key == 27:
        break
    elif key >= 49 and key <= 57:
        scale_factor = d[key]
        detection_level = key - 48

    print(f"\rDetection Level: {detection_level} Scale Factor: {scale_factor}", end="")

print("\nClosing...")
video_capture.release()
cv2.destroyAllWindows()
