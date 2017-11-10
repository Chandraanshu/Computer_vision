import numpy as np
import math
import cv2


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
