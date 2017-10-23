import numpy as np
from scipy.linalg import svd
import video_io
import collections
import cv2


def applyHomography(originalImage, homographyMatrix):
    """Transforms the image according to a homography.

    Note that the resultant picture would likely not fit in the same area as the
    previous, so the result will be cropped accordingly.

    Args:
        originalImage: The image to be transformed.
        homographyMatrix: The homography matrix to be applied to the image.

    Returns:
        The transformed image.
    """
    numRows = originalImage.shape[0]
    numCols = originalImage.shape[1]

    # Form array of point coordinates.
    imagePoints = []
    for i in range(numRows):
        for j in range(numCols):
            imagePoints.append([i, j, 1])
    imagePoints = np.array(imagePoints).transpose()

    # Compute positions of points after applying the homography matrix.
    transformedImage = np.matmul(homographyMatrix, imagePoints)
    transformedImage[np.abs(transformedImage) < 1e-15] = 0
    assert np.all(np.abs(transformedImage[2] - 1) < 1e-10)

    # Defaultdict(list) has a default value of []
    # newMapping maps the transformed points to pixel values.
    # One point can have multiple values due to rounding.
    newMapping = collections.defaultdict(list)

    for i in range(numRows * numCols):
        newPoint = tuple(np.round(transformedImage[:2, i]).astype(int))
        oldPoint = (i // numCols, i % numCols)
        newMapping[newPoint].append(originalImage[oldPoint])

    newImage = np.zeros(originalImage.shape)

    for i in range(numRows):
        for j in range(numCols):
            # Map each coordinate in the new image to an average of all pixel
            # values assigned to that coordinate under newMapping.
            # If no such mapping was specified, make the pixel white.
            newImage[i, j] = np.average(newMapping.get((i, j), [[255, 255, 255]]), axis=0)

    return np.array(newImage).astype(np.uint8)


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


if __name__ == '__main__':
    frame = video_io.readVideo('traffic.mp4')[0]
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    originalPoints = np.array([
        [0, 0],
        [0, frameWidth],
        [frameHeight, frameWidth],
        [frameHeight, 0],
    ])
    finalPoints = np.array([
        [0, frameHeight],
        [frameWidth, frameHeight],
        [frameWidth, 0],
        [0, 0],
    ])

    homography = computeHomography(originalPoints, finalPoints)

    transformedImage = applyHomography(frame, homography)
    cv2.imshow('Frame', transformedImage)
    cv2.waitKey(10000)
