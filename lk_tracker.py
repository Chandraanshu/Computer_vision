import cv2
import numpy as np
from scipy import signal
import math


WINDOW_SIZE = 13  # Must be an odd number

def imageShrink(image, size):
    """
    Reduces by half, uses gausian filter
    size :
        size of Gauss kernel to use
    """
    gkern = np.outer(signal.gaussian(size, 2.5), signal.gaussian(size, 2.5))
    total = gkern.sum()
    gkern = gkern/total     #Sum of Gauss kernel must be one.... Doesn't work without it
    b = signal.convolve2d(image[:,:,0], gkern, 'same')
    g = signal.convolve2d(image[:,:,1], gkern, 'same')
    r = signal.convolve2d(image[:,:,2], gkern, 'same')
    shrunk = np.dstack([b,g,r])
    shrunk = (shrunk).astype(np.uint8)
    return shrunk[::2,::2]

def imageExpand(image, size):
    gkern = np.outer(signal.gaussian(size, 2.5), signal.gaussian(size, 2.5))
    total = gkern.sum()
    gkern = gkern/total
    expand = np.zeros((image.shape[0]*2, image.shape[1]*2, 3), dtype=np.float64)
    expand[::2,::2] = image[:,:,:]
    b = signal.convolve2d(expand[:,:,0], gkern, 'same')
    g = signal.convolve2d(expand[:,:,1], gkern, 'same')
    r = signal.convolve2d(expand[:,:,2], gkern, 'same')
    expand = np.dstack([b,g,r])
    expand = expand.astype(np.uint8)
    return expand

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


def LKTrackerFrameToFrame(frame1, frame2, pixelCoords, windowSize):
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
    print(pixelCoords)
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


def LKTracker(cap, pixelCoords, windowSize):
    """Tracks a pixel from frame to frame using the Lucas-Kanade Method.

    In fact, tracks a square window centred at the pixel.
    The smaller the difference from frame to frame, the better the tracking.
    Returns a generator object that can be iterated over to give successive
    positions of the pixel.

    Args:
        cap: cv2.VideoCapture object representing the video in which to track
            the pixel.
        pixelCoords: Numpy array with shape (2, ) giving coordinates of pixel.
        windowSize: The side length of the square window being tracked.

    Yields:
        The current frame, and the coordinates of the pixel in that frame. The
        first value yielded is the original position.
    """
    # Get first frame.
    _, frameNew = cap.read()
    yield frameNew, pixelCoords

    while cap.isOpened():
        frameOld = frameNew
        ret, frameNew = cap.read()
        if ret:
            pixelCoords = np.round(LKTrackerFrameToFrame(frameOld,
                                            frameNew,
                                            pixelCoords,
                                            windowSize)).astype(int)
            yield frameNew, pixelCoords

def drawRectangleOnImage(image, centre, width, height, color):
    cv2.rectangle(image, (centre[0] - width // 2, centre[1] - height // 2), (centre[0] + width // 2, centre[1] + height // 2), color=color)

def generateShrinkPyramid(frame, depth):
    '''
    Generates images half in size, returns as list
    depth :
        how many levels deep to go
    '''
    shrunkImages = []
    window = 7          #NEED TO FINETUNE GAUSS KERNEL SIZE
    for i in range(depth):
        if i == 0:
            shrunkImages.append(imageShrink(frame, window))
            continue
        shrunkImages.append(imageShrink(shrunkImages[i-1], window))
        newWindow = math.ceil(window/2)      #Gauss window size needs to reduce as the image gets smaller, else the blurring is excessive
        if newWindow % 2 == 0 : newWindow = newWindow + 1
        window = newWindow
    return shrunkImages

def test(cap, pixelCoords, windowSize):
    _, frameNew = cap.read()
    print(pixelCoords)
    tempCoords = pixelCoords         #3 levels(207 and 170 div by 8)
    yield frameNew, pixelCoords

    while cap.isOpened():
        frameOld = frameNew
        ret, frameNew = cap.read()
        tempCoords = np.array(np.floor_divide(tempCoords, 8))
        shrunkImagesOld = generateShrinkPyramid(frameOld, 3)
        shrunkImagesNew = generateShrinkPyramid(frameNew, 3)
        if ret:
            print('ONCE')
            for i in reversed(range(3)):
                print(tempCoords)
                tempCoords = np.round(LKTrackerFrameToFrame(shrunkImagesOld[i],
                                                shrunkImagesNew[i],
                                                tempCoords,
                                                windowSize)).astype(int)
                tempCoords = np.array(np.multiply(tempCoords,2))
        yield frameNew, tempCoords

if __name__ == '__main__':
    cap = openVideo('traffic.mp4')

    # Set up to track top of yellow taxi in traffic.mp4.
    #for frame, coords in LKTracker(cap, np.array([207, 170]), WINDOW_SIZE):
    for frame, coords in test(cap, np.array([207, 170]), WINDOW_SIZE):
        drawRectangleOnImage(frame, coords, WINDOW_SIZE, WINDOW_SIZE, (0, 0, 255))
        cv2.imshow('Frame', frame)
        cv2.waitKey(50)

    #_, frameNew = cap.read()
    #shrunkImages = generateShrinkPyramid(frameNew, 3)
    #a = imageExpand(shrunkImages[0], 7)
    #cv2.imshow('Frame',a)
    #cv2.waitKey(5000)
    shutdown(cap)
