import cv2
import numpy as np
from scipy import signal
import math


WINDOW_SIZE = 13  # Must be an odd number


def openVideo(fileName):
    """Opens video stored at the given file.

    Args:
        fileName: The name of the video file.

    Returns:
        A cv2.VideoCapture object.

    Throws:
        IOError: If there are issues loading the video from file.
    """
    cap = cv2.VideoCapture(fileName)

    if not cap.isOpened():
        raise IOError("Error opening video file.")

    return cap


def shutdown(*caps):
    """Shuts down all cv2.VideoCapture objects passed in and closes all windows.

    Args:
        *caps: Any number of cv2.VideoCapture objects passed in as multiple
            arguments.
    """
    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()


def pixelDiffImages(img1, img2, x, y, width, height):
    """Computes the pixel-wise difference between Lab* images.

    Requires images to be using the Lab* colour encoding scheme.
    Images should be passed in as numpy arrays.
    The difference algorithm used is Delta E (CIE 1976).
    Difference is only computed for window whose top left corner is at (x, y)
    and has width and height as given.

    Args:
        img1: Numpy array containing first image.
        img2: Numpy array containing second image.
        x: x coordinate of top left corner of window.
        y: y coordinate of top left corner of window.
        width: Width of window.
        height: Height of window.

    Returns:
        A numpy array with shape (width, height) containing the pixel-wise
        difference between the given images.
    """
    diff = img1[x : x + width, y : y + height] - img2[x : x + width, y : y + height]
    return np.array([[math.sqrt(np.linalg.norm(x)) for x in row] for row in diff])


def LKTracker(frame1, frame2, pixelCoords, windowSize):
    """Tracks a pixel from one frame to another using the Lucas-Kanade Method.

    In fact, tracks a square window centred at the pixel.
    The smaller the difference between the two frames, the better the tracking.
    Requires images to be using the BGR colour encoding scheme.

    Args:
        frame1: The original frame.
        frame2: The new frame.
        pixelCoords: Numpy array with shape (2, ) giving coordinates of pixel.
        windowSize: The side length of the square window being tracked.

    Returns:
        The computed position of the pixel in the new frame, as a numpy array of
        x and y coordinates.
    """
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2Lab)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2Lab)

    # Get top left corner of window.
    topLeftX, topLeftY = pixelCoords - windowSize // 2

    # Compute horizontal and vertical gradients for the original frame.
    gx = pixelDiffImages(frame1[:, 1 : ],
                         frame1[:, : -1],
                         topLeftX,
                         topLeftY,
                         windowSize,
                         windowSize)
    gy = pixelDiffImages(frame1[1 :, :],
                         frame1[ : -1, :],
                         topLeftX,
                         topLeftY,
                         windowSize,
                         windowSize)

    # Compute difference between original and new frames.
    diff = pixelDiffImages(frame1,
                           frame2,
                           topLeftX,
                           topLeftY,
                           windowSize,
                           windowSize)

    # Compute components of Harris matrix.
    Ixx = gx ** 2
    Iyy = gy ** 2
    Ixy = gx * gy

    # Compute Gaussian kernel for weighting pixels in the window.
    gkern = np.outer(signal.gaussian(windowSize, 2.5),
                     signal.gaussian(windowSize, 2.5))

    # Construct matrices and solve the matrix-vector equation to get the
    # movement of the pixel.
    Z = np.array([[np.sum(Ixx * gkern), np.sum(Ixy * gkern)],
                  [np.sum(Ixy * gkern), np.sum(Iyy * gkern)]])
    b = np.array([np.sum(diff * gx * gkern), np.sum(diff * gy * gkern)])
    d = np.linalg.solve(Z, b)

    # Compute new position of pixel
    return pixelCoords + d


if __name__ == '__main__':
    cap = openVideo('traffic.mp4')

    _, frame1 = cap.read()
    _, frame2 = cap.read()

    # Set up to track top of yellow taxi in traffic.mp4.
    print(LKTracker(frame1, frame2, np.array([207, 170]), WINDOW_SIZE))

    # cv2.rectangle(frame, (200, 165), (215, 175), color=(0, 0, 255))

    shutdown(cap)
