import cv2
import numpy as np
from scipy import signal
import math
import video_io
import utils
import constants
import time


def generateShrinkPyramid(image, depth):
    """Generates an image pyramid, composed of progressively smaller images.

    First image in the list is the original one.

    Args:
        image: The image for which the pyramid will be constructed.
        depth: Total number of images in the pyramid.

    Returns:
        List of images, represented as numpy arrays.
    """
    shrunkImages = [image]
    window = constants.GAUSS_BLUR_WINDOW_SIZE  # TODO: Finetune Gaussian kernel size.
    for i in range(depth - 1):
        shrunkImages.append(utils.imageShrink(shrunkImages[-1], window))
        
        # Gauss window size needs to reduce as the image gets smaller,
        # else the blurring is excessive.
        window = window // 2
        if window % 2 == 0:
            window += 1

    return shrunkImages


def LKTrackerImageToImage(imageOld, pixelCoordsOld, imageNew,
                          pixelCoordsNew, windowSize):
    """Tracks a pixel from one image to another using the Lucas-Kanade Method.

    In fact, tracks a square window centred at the pixel.
    The smaller the difference between the two frames, the better the tracking.
    Requires images to be using the BGR colour encoding scheme.

    Args:
        imageOld: The original image.
        pixelCoordsOld: Numpy array with shape (2, ) giving original coordinates
            of pixel in imageOld.
        imageNew: The new image.
        pixelCoordsNew: Numpy array with shape (2, ) giving best approximation for
            coordinates of pixel in imageNew.
        windowSize: The side length of the square window being tracked.
    Returns:
        The computed position of the pixel in the new frame, as a numpy array of
        x and y coordinates.
    """
    # imageOld = cv2.cvtColor(imageOld, cv2.COLOR_BGR2GRAY)
    # imageNew = cv2.cvtColor(imageNew, cv2.COLOR_BGR2GRAY)

    # Get top left corner of window.

    topLeftX1, topLeftY1 = pixelCoordsOld - windowSize // 2
    topLeftX2, topLeftY2 = pixelCoordsNew - windowSize // 2

    # Compute horizontal and vertical gradients for the original frame.
    gx = utils.pixelDiffImages(imageOld,
                               topLeftX1,
                               topLeftY1 + 1,
                               imageOld,
                               topLeftX1,
                               topLeftY1,
                               windowSize,
                               windowSize)
    
    gy = utils.pixelDiffImages(imageOld,
                               topLeftX1 + 1,
                               topLeftY1,
                               imageOld,
                               topLeftX1,
                               topLeftY1,
                               windowSize,
                               windowSize)
    
    # Compute difference between original and new frames.
    diff = utils.pixelDiffImages(imageOld,
                                 topLeftX1,
                                 topLeftY1,
                                 imageNew,
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
    return pixelCoordsNew + d[: : -1]


def LKTrackerFrameToFrame(frameOld, frameNew, pixelCoords,
                          windowSize, pyramidDepth):
    """Tracks a pixel from frame to frame using Lucas-Kanade over a pyramid.

    In fact, tracks a square window centred at the pixel.

    Args:
        frameOld: The original frame.
        frameNew: The updated frame.
        pixelCoords: Numpy array with shape (2, ) giving coordinates of the
            pixel being tracked.
        windowSize: The side length of the square window being tracked.
        pyramidDepth: The depth of the pyramid being used.

    Returns:
        Numpy array with shape (2, ) giving the updated coordinates of the
        pixel being tracked.
    """
    shrunkImagesOld = generateShrinkPyramid(frameOld, pyramidDepth)
    shrunkImagesNew = generateShrinkPyramid(frameNew, pyramidDepth)

    # Best estimate in smallest pyramid is original position, scaled down.
    newCoords = pixelCoords // (2 ** (pyramidDepth - 1))

    for i in reversed(range(pyramidDepth)):
        newCoords = LKTrackerImageToImage(shrunkImagesOld[i],
                                          pixelCoords // (2 ** i),
                                          shrunkImagesNew[i],
                                          newCoords,
                                          windowSize)

        newCoords = np.round(newCoords * 2).astype(int)

    return newCoords // 2


if __name__ == '__main__':
    video = video_io.readVideo(constants.LK_TRACKER_DEMO_VIDEO, constants.NUM_FRAMES_TO_TRACK)
    video = np.transpose(video, [0, 2, 1, 3]).copy()

    # Convert from image coordinates to matrix coordinates.
    pixelToTrack = constants.PIXEL_TO_TRACK[: : -1]
    pixelPositions = [pixelToTrack]
    for frameIdx in range(min(constants.NUM_FRAMES_TO_TRACK, len(video) - 1)):
        oldFrame = cv2.cvtColor(video[frameIdx], cv2.COLOR_BGR2GRAY)
        newFrame = cv2.cvtColor(video[frameIdx + 1], cv2.COLOR_BGR2GRAY)
        pixelToTrack = LKTrackerFrameToFrame(oldFrame,
                                             newFrame,
                                             pixelToTrack,
                                             constants.TRACK_WINDOW_SIZE,
                                             constants.PYRAMID_DEPTH)
        pixelPositions.append(pixelToTrack)

    for i, pixelPosition in enumerate(pixelPositions):
        utils.drawRectangleOnImage(video[i],
                                   pixelPosition[: : -1],
                                   constants.TRACK_WINDOW_SIZE,
                                   constants.TRACK_WINDOW_SIZE,
                                   (0, 0, 255))
    
    video_io.displayVideo(video, FPS=5)

    videoWriter = video_io.createVideoWriter(video[0], constants.LK_TRACKER_VIDEO)
    for frame in video:
        video_io.writeFrame(videoWriter, frame)
        time.sleep(0.01)

