import cv2
import numpy as np

cap = cv2.VideoCapture('traffic.mp4')

if not cap.isOpened():
    print("Error opening video file.")

print("Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Number of frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()
