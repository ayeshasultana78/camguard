import cv2
import numpy as np
import time

# 1. Load Classifiers (Ensure these XML files are in your 'stats/Haarcascades/' folder)
face_cascade = cv2.CascadeClassifier('stats/Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('stats/Haarcascades/haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('stats/Haarcascades/haarcascade_fullbody.xml')
car_cascade = cv2.CascadeClassifier('stats/Haarcascades/haarcascade_car.xml')

# 2. Function for Real-time Face and Eye Detection
def detect_features(gray, frame):
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame

# 3. Live Webcam Detection Loop
video = cv2.VideoCapture(0)

print("Press 'q' to quit the video stream.")

while True:
    check, frame = video.read()
    if not check:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_features(gray, frame)
    
    cv2.imshow('Camguard - Smart CCTV Feed', canvas)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# 4. Static Image Processing Example (Modi.jpg)
# Note: Ensure the path matches your repo structure
test_img = cv2.imread(r'stats\image_examples\Modi.jpg', 1)
if test_img is not None:
    resized = cv2.resize(test_img, (500, 500))
    # Detect faces in static image
    gray_static = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces_static = face_cascade.detectMultiScale(gray_static, 1.05, 5)
    
    for (x, y, w, h) in faces_static:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 100), 2)
        
    cv2.imshow('Static Detection Test', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
