import os
import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Suppress TensorFlow Lite Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Webcam settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Change to 1 if external webcam is used
cap.set(3, wCam)
cap.set(4, hCam)
time.sleep(2)  # Allow camera to warm up
pTime = 0

# Initialize Hand Detector
detector = htm.HandDetector(detectionCon=0.7, trackCon=0.5)

# Audio control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture image from webcam.")
        continue  # Skip iteration if no image is captured

    # Find hand landmarks
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Get the tip coordinates of thumb and index finger
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw markers
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between the fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Map hand range (50-300) to volume range (-65 to 0 dB)
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        # Set the system volume
        volume.SetMasterVolumeLevel(vol, None)

        # Visual feedback for low distance
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Volume Control", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
