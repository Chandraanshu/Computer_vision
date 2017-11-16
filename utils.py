import numpy as np
import math
import cv2
from scipy import signal
import constants


def pixelDiffImages(img1, x1, y1, img2, x2, y2, width, height):
    """Computes the pixel-wise difference between 2 images.
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
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return (img1[x1 : x1 + width, y1 : y1 + height] -
            img2[x2 : x2 + width, y2 : y2 + height])


def drawRectangleOnImage(image, centre, width, height, color):
    """Draws a rectangle on the given image.

    Args:
        image: The image on which the rectangle is to be drawn.
        centre: The coordinates of the centre of the rectangle.
        width: The width of the rectangle.
        height: The height of the rectangle.
        color: The color of the rectangle.
    """
    cv2.rectangle(image,
                  (centre[0] - width // 2, centre[1] - height // 2),
                  (centre[0] + width // 2, centre[1] + height // 2),
                  color=color)


def imageExpand(image, size):
    gkern = np.outer(signal.gaussian(size, constants.GAUSS_STDDEV), signal.gaussian(size, constants.GAUSS_STDDEV))
    gkern = gkern / gkern.sum()

    # Expand each pixel into 4 pixels.
    expanded = np.repeat(np.repeat(image, 2, axis=1), 2, axis=0)
    return signal.fftconvolve(expanded, gkern, 'same').astype(np.uint8)


def imageShrink(image, size):
    """Reduces the size of an image by half using a Gaussian filter.

    Args:
        image: Numpy array containing the image to be shrunk.
        size: Size of the Gaussian filter to use.

    Returns:
        Shrunk image as an np array.
    """
    gkern = np.outer(signal.gaussian(size, constants.GAUSS_STDDEV), signal.gaussian(size, constants.GAUSS_STDDEV))
    gkern = gkern / np.sum(gkern)  # Normalize sum of kernel to be 1.
    return signal.fftconvolve(image, gkern, 'same').astype(np.uint8)[ : : 2, : : 2]


def cropImage(image, topMargin, bottomMargin, leftMargin, rightMargin):
    return image[topMargin : image.shape[0] - bottomMargin, leftMargin : image.shape[1] - rightMargin]


def drawTopLeftRectangleOnImage(image, topLeft, width, height, color):
    """Draws a rectangle on the given image.

    Args:
        image: The image on which the rectangle is to be drawn.
        centre: The coordinates of the centre of the rectangle.
        width: The width of the rectangle.
        height: The height of the rectangle.
        color: The color of the rectangle.
    """
    cv2.rectangle(image,
                  (topLeft[0], topLeft[1]),
                  (topLeft[0] + width, topLeft[1] + height),
                  color=color)
