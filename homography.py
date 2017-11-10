import numpy as np
from scipy.linalg import svd
import video_io
import collections
import cv2
import time
import utils


def computeHomography(originalPoints, finalPoints):
    """Compute homography matrix transforming one plane into another.

    Needs at least 4 corresponding points on each plane to find the homography.

    Args:
        originalPoints: A numpy array with shape (numPoints, 2), representing
            the coordinates of the points in the first plane.
        finalPoints: A numpy array with shape (numPoints, 2), representing
            the coordinates of the points in the second plane.

    Returns:
        A homography matrix, in the form of a 3x3 numpy array. The last element
        of the matrix is normalized to 1.
    """
    equationMatrix = np.zeros([originalPoints.shape[0] * 2, 9])

    for i, (originalPoint, finalPoint) in enumerate(zip(originalPoints,
                                                        finalPoints)):
        # Add a 1 to the original point.
        homogenizedOriginalPoint = np.concatenate((originalPoint, [1]))

        equationMatrix[i * 2, 0 : 3] = homogenizedOriginalPoint
        equationMatrix[i * 2, 6 : 9] = -finalPoint[0] * homogenizedOriginalPoint
        equationMatrix[i * 2 + 1, 3 : 6] = homogenizedOriginalPoint
        equationMatrix[i * 2 + 1, 6 : 9] = -finalPoint[1] * homogenizedOriginalPoint

    U, S, Vt = svd(equationMatrix)

    # Last row of V' gives the solution.
    solution = Vt[-1]

    # Last element must not be close to 0, since we're normalizing using that.
    assert abs(solution[-1]) > 1e-15

    solution /= solution[-1]

    # Set very small values to 0.
    solution[np.abs(solution) < 1e-15] = 0

    return np.reshape(solution, (3, 3))


def computeMapping(imageHeight, imageWidth, homographyMatrix):
    """Computes a mapping from old coordinates to new, under a given homography.

    Only points for which the new coordinates lie within the bounds of the
    original image are kept.

    Args:
        imageHeight: Height of the image.
        imageWidth: Width of the image.
        homographyMatrix: A matrix representing the homography which transforms
            the image.

    Returns:
        Coordinates of the new and old points. Each of these is returned as a
        numpy array with shape (2, numPoints), giving x and y coordinates in
        separate rows.
    """
    # Form array of point coordinates.
    imagePoints = [[i, j, 1] for i in range(imageHeight) for j in range(imageWidth)]
    imagePoints = np.array(imagePoints).T

    # Compute positions of points after applying the homography matrix.
    newPositions = np.matmul(homographyMatrix, imagePoints)
    newPositions[np.abs(newPositions) < 1e-15] = 0  # Set small values to 0
    newPositions = np.round((newPositions / newPositions[-1])[:2]).astype(int)

    # Construct a mapping from transformed points to the original points.
    mapping = {}
    for i in range(imageHeight * imageWidth):
        x, y = newPositions[:, i]
        if 0 <= x and x < imageHeight and 0 <= y and y < imageWidth:
            mapping[x, y] = (i // imageWidth, i % imageWidth)

    # Convert the mapping into arrays of coordinates
    newPoints, oldPoints = mapping.keys(), mapping.values()
    newPoints, oldPoints = list(map(list, newPoints)), list(map(list, oldPoints))
    return np.array(newPoints).T, np.array(oldPoints).T


def transformImage(originalImage, oldPoints, newPoints):
    """Transforms an image mapping its original points to new positions.

    Fills in the new image with white pixels wherever there is no mapping.
    The original image must be in BGR.

    Args:
        originalImage: The image to be transformed.
        oldPoints: A numpy array with shape (2, numPoints). The first row
            contains the x coordinates of the points to be mapped, while the
            second row contains the y coordinates.
        newPoints: A numpy array with shape (2, numPoints). The first row
            contains the new x coordinates of the points being mapped, while the
            second row contains the new y coordinates.

    Returns:
        The transformed image.
    """
    # originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    newImage = np.full(originalImage.shape, fill_value=255, dtype=np.uint8)
    newImage[newPoints[0], newPoints[1]] = originalImage[oldPoints[0], oldPoints[1]]

    return newImage


if __name__ == '__main__':
    video = video_io.readVideo('Shadow.mp4')

    # video = video[:, :, 500:-300]

    video[np.any(video > 40, axis=3)] = 255
    # video_io.displayVideo(video)

    frame = video[0]

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    originalPoints = np.array([
        [100, 10],
        [600, 30],
        [frameHeight - 100, frameWidth - 50],
        [80, frameWidth - 50],
    ])
    finalPoints = np.array([
        [100, 100],
        [400, 100],
        [400, 600],
        [100, 600],
    ])
    homographyMatrix = computeHomography(originalPoints, finalPoints)
    newPoints, oldPoints = computeMapping(frameHeight, frameWidth, homographyMatrix)

    # for point in originalPoints:
    #     utils.drawRectangleOnImage(frame,
    #                          list(point)[::-1],
    #                          6,
    #                          6,
    #                          (0, 0, 255))

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 600,600)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(3000)

    # originalPoints = [list(point) + [1] for point in originalPoints]
    # originalPoints = np.array(originalPoints).transpose()
    # solution = np.matmul(homography, originalPoints)
    # print(solution / solution[-1])

    for frame in video:
        transformedFrame = transformImage(frame, oldPoints, newPoints)
        # transformedFrame[450:] = 255
        # transformedFrame[:,300:] = 255
        cv2.imshow('Frame', transformedFrame)
        cv2.waitKey(1)
