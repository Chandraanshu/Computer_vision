import cv2
import numpy as np
from scipy import signal
import math
import video_io


WINDOW_SIZE = 13  # Must be an odd number
PYRAMID_DEPTH = 3


def imageShrink(image, size):
    """
    Reduces by half, uses gausian filter
    size :
        size of Gauss kernel to use
    """
    gkern = np.outer(signal.gaussian(size, 2.5), signal.gaussian(size, 2.5))
    total = np.sum(gkern)
    gkern = gkern / total     #Sum of Gauss kernel must be one.... Doesn't work without it
    b = signal.fftconvolve(image[ : , : , 0], gkern, 'same')
    g = signal.fftconvolve(image[ : , : , 1], gkern, 'same')
    r = signal.fftconvolve(image[ : , : , 2], gkern, 'same')
    return np.dstack([b,g,r]).astype(np.uint8)[ : : 2, : : 2]


def generateShrinkPyramid(frame, depth):
    '''
    Generates images half in size, returns as list
    depth :
        how many levels deep to go
    '''
    shrunkImages = []
    window = 11         #NEED TO FINETUNE GAUSS KERNEL SIZE
    for i in range(depth):
        if i == 0:
            shrunkImages.append(imageShrink(frame, window))
            continue
        shrunkImages.append(imageShrink(shrunkImages[i-1], window))
        newWindow = math.ceil(window/2)      #Gauss window size needs to reduce as the image gets smaller, else the blurring is excessive
        if newWindow % 2 == 0 : newWindow = newWindow + 1
        window = newWindow
    return shrunkImages


def imageExpand(image, size):
    gkern = np.outer(signal.gaussian(size, 2.5), signal.gaussian(size, 2.5))
    total = gkern.sum()
    gkern = gkern/total
    expand = np.zeros((image.shape[0]*2, image.shape[1]*2, 3), dtype=np.float64)
    expand[::2,::2] = image[:,:,:]
    b = signal.fftconvolve(expand[:,:,0], gkern, 'same')
    g = signal.fftconvolve(expand[:,:,1], gkern, 'same')
    r = signal.fftconvolve(expand[:,:,2], gkern, 'same')
    expand = np.dstack([b,g,r])
    expand = expand.astype(np.uint8)
    return expand


def pixelDiffImages(img1, x1, y1, img2, x2, y2, width, height):
    """Computes the pixel-wise difference between Lab* images.

    Requires images to be using the Lab* colour encoding scheme.
    Images should be passed in as numpy arrays.
    The difference algorithm used is Delta E (CIE 1976).
    Difference is only computed for window whose top left corner is at (x, y)
    and has width and height as given.

    Args:
        img1: Numpy array containing first image.
        x1: x coordinate of top left corner of window in image1.
        y1: y coordinate of top left corner of window in image1.
        img2: Numpy array containing second image.
        x2: x coordinate of top left corner of window in image2.
        y2: y coordinate of top left corner of window in image2.
        width: Width of window.
        height: Height of window.

    Returns:
        A numpy array with shape (width, height) containing the pixel-wise
        difference between the given images.
    """
    diff = (img1[x1 : x1 + width, y1 : y1 + height] -
            img2[x2 : x2 + width, y2 : y2 + height])
    return np.array(
        [[math.sqrt(np.linalg.norm(x)) for x in row] for row in diff]
    )


def LKTrackerImageToImage(image1, pixelCoords1, image2,
                          pixelCoords2, windowSize):
    """Tracks a pixel from one image to another using the Lucas-Kanade Method.

    In fact, tracks a square window centred at the pixel.
    The smaller the difference between the two frames, the better the tracking.
    Requires images to be using the BGR colour encoding scheme.

    Args:
        image1: The original frame.
        pixelCoords1: Numpy array with shape (2, ) giving original coordinates
            of pixel (in image1).
        image2: The new frame.
        pixelCoords2: Numpy array with shape (2, ) giving best approximation for
            coordinates of pixel in image2.
        windowSize: The side length of the square window being tracked.

    Returns:
        The computed position of the pixel in the new frame, as a numpy array of
        x and y coordinates.
    """
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab)

    # Get top left corner of window.
    topLeftX1, topLeftY1 = pixelCoords1 - windowSize // 2
    topLeftX2, topLeftY2 = pixelCoords2 - windowSize // 2

    # Compute horizontal and vertical gradients for the original frame.
    gx = pixelDiffImages(image1[:, 1 : ],
                         topLeftX1,
                         topLeftY1,
                         image1[:, : -1],
                         topLeftX1,
                         topLeftY1,
                         windowSize,
                         windowSize)
    gy = pixelDiffImages(image1[1 :, :],
                         topLeftX1,
                         topLeftY1,
                         image1[ : -1, :],
                         topLeftX1,
                         topLeftY1,
                         windowSize,
                         windowSize)

    # Compute difference between original and new frames.
    diff = pixelDiffImages(image1,
                           topLeftX1,
                           topLeftY1,
                           image2,
                           topLeftX2,
                           topLeftY2,
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
    return pixelCoords2 + d


def LKTrackerFrameToFrame(frameOld, frameNew, pixelCoords,
                          windowSize, pyramidDepth):
    shrunkImagesOld = generateShrinkPyramid(frameOld, pyramidDepth)
    shrunkImagesNew = generateShrinkPyramid(frameNew, pyramidDepth)

    # Best estimate in smallest pyramid is original position, scaled down.
    newCoords = pixelCoords // 2 ** pyramidDepth

    for i in reversed(range(pyramidDepth)):
        newCoords = LKTrackerImageToImage(shrunkImagesOld[i],
                                          pixelCoords // 2 ** (i + 1),
                                          shrunkImagesNew[i],
                                          newCoords,
                                          windowSize)
        newCoords = np.round(newCoords * 2).astype(int)

    return newCoords


def drawRectangleOnImage(image, centre, width, height, color):
    cv2.rectangle(image, (centre[0] - width // 2, centre[1] - height // 2), (centre[0] + width // 2, centre[1] + height // 2), color=color)


if __name__ == '__main__':
    video = video_io.readVideo('traffic.mp4')

    # Set up to track top of yellow taxi in traffic.mp4.
    pixelToTrack = np.array([207, 170])
    for frameIdx in range(len(video) - 1):
        pixelToTrack = LKTrackerFrameToFrame(video[frameIdx],
                                             video[frameIdx + 1],
                                             pixelToTrack,
                                             WINDOW_SIZE,
                                             PYRAMID_DEPTH)
        drawRectangleOnImage(video[frameIdx + 1],
                             pixelToTrack,
                             WINDOW_SIZE,
                             WINDOW_SIZE,
                             (0, 0, 255))

    video_io.displayVideo(video, FPS=5)
