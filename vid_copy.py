import cv2
import numpy as np
import os

ORIGINAL_FILE = 'traffic.mp4'
COPY_FILE = 'trafficcopy.mp4'

cap = cv2.VideoCapture(ORIGINAL_FILE)

if not cap.isOpened():
    print("Error opening video file.")

try:
    os.remove(COPY_FILE)
except FileNotFoundError:
    pass

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Since fps is only 10, the copy will run slowly.
out = cv2.VideoWriter(COPY_FILE,
                      apiPreference=cv2.CAP_ANY,
                      fourcc=cv2.VideoWriter_fourcc('M','J','P','G'),
                      fps=10,
                      frameSize=(frameWidth, frameHeight))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
