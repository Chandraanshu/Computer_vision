import cv2
import numpy as np
from scipy import signal
import math

WINDOW_X = 200
WINDOW_Y = 165
WINDOW_WIDTH = 15
WINDOW_HEIGHT = 10

cap = cv2.VideoCapture('traffic.mp4')

if not cap.isOpened():
    print("Error opening video file.")

# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
#         cv2.rectangle(frame, (200, 200), (215, 215), color=(0, 0, 255))
#         # Display the resulting frame
#         cv2.imshow('Frame', frame)
#
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break

_, frame1 = cap.read()
_, frame2 = cap.read()

# cv2.rectangle(frame, (200, 165), (215, 175), color=(0, 0, 255))
labframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2Lab)
labframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2Lab)

def pixelDiffImages(mat1, mat2, x, y, width, height):
    """Computes the pixel-wise difference between Lab* images.

    Requires images to be using the Lab* colour encoding scheme.
    Images should be passed in as numpy arrays.
    The difference algorithm used is Delta E (CIE 1976).
    Difference is only computed for window whose top left corner is at (x, y)
    and has width and height as given.

    Args:
        mat1: Numpy array containing first image.
        mat2: Numpy array containing second image.
        x: x coordinate of top left corner of window.
        y: y coordinate of top left corner of window.
        width: Width of window.
        hieght: Hieght of window.

    Returns:
        A numpy array with shape (width, height) containing the pixel-wise
        difference between the given images.
    """
    diff = mat1[x : x + width, y : y + height] - mat2[x : x + width, y : y + height]
    return np.array([[math.sqrt(np.linalg.norm(x)) for x in row] for row in diff])

gx = pixelDiffImages(labframe1[:, 1 : ], labframe1[:, : -1], WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
gy = pixelDiffImages(labframe1[1 :, :], labframe1[ : -1, :], WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
diff = pixelDiffImages(labframe1, labframe2, WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

Ixx = gx ** 2
Iyy = gy ** 2
Ixy = gx * gy

gkern = np.outer(signal.gaussian(WINDOW_WIDTH, 2.5), signal.gaussian(WINDOW_HEIGHT, 2.5))

Z = np.array([[np.sum(Ixx * gkern), np.sum(Ixy * gkern)], [np.sum(Ixy * gkern), np.sum(Iyy * gkern)]])
b = np.array([np.sum(diff * gx * gkern), np.sum(diff * gy * gkern)])
d = np.linalg.solve(Z, b)
print(Z)
print(b)
print(d)

# cv2.imshow('Image', frame1)
# cv2.waitKey(2000)

cap.release()

cv2.destroyAllWindows()
