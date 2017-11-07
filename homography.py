import numpy as np
from scipy.linalg import svd
import video_io
import collections
import cv2
import time


def applyHomography(originalImage, homographyMatrix, newPositions):
    """Transforms the image according to a homography.

    Note that the resultant picture would likely not fit in the same area as the
    previous, so the result will be cropped accordingly.

    Args:
        originalImage: The image to be transformed.
        homographyMatrix: The homography matrix to be applied to the image.

    Returns:
        The transformed image.
    """
    start = time.time()
    numRows = originalImage.shape[0]
    numCols = originalImage.shape[1]
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    # Defaultdict(list) has a default value of []
    # newMapping maps the transformed points to pixel values.
    # One point can have multiple values due to rounding.
    print('Done transforming')
    # newMapping = collections.defaultdict(list)
    newMapping = dict()

    # newImage = np.full(originalImage.shape, 255)

    for i in range(numRows * numCols):
        x, y = newPositions[:, i]
        newMapping[x, y] = originalImage[i // numCols, i % numCols]

    print('Done mapping')

    newImage = np.empty(originalImage.shape)

    for i in range(numRows):
        for j in range(numCols):
            # If no such mapping was specified, make the pixel white.
            newImage[i, j] = newMapping.get((i, j), 255)

    print('Done changing')
    print(time.time() - start)
    return newImage.astype(np.uint8)


def computeNewPositions(imageHeight, imageWidth, homographyMatrix):
    # Form array of point coordinates.
    imagePoints = []
    inner_range = range(imageWidth)
    for i in range(imageHeight):
        for j in range(imageWidth):
            imagePoints.append([i, j, 1])
    imagePoints = np.array(imagePoints).transpose()

    # Compute positions of points after applying the homography matrix.
    transformedImage = np.matmul(homographyMatrix, imagePoints)
    transformedImage[np.abs(transformedImage) < 1e-15] = 0
    return np.round((transformedImage / transformedImage[-1])[:2])


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


if __name__ == '__main__':
    video = video_io.readVideo('shadowTrim.mp4')[320:400]

    video = video[:, :, 500:-300]

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
    newPositions = computeNewPositions(frameHeight, frameWidth, homographyMatrix)

    # for point in originalPoints:
    #     drawRectangleOnImage(frame,
    #                          list(point)[::-1],
    #                          6,
    #                          6,
    #                          (0, 0, 255))

    #cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Frame', 600,600)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(3000)

    # originalPoints = [list(point) + [1] for point in originalPoints]
    # originalPoints = np.array(originalPoints).transpose()
    # solution = np.matmul(homography, originalPoints)
    # print(solution / solution[-1])

    for frame in video:
        transformedFrame = applyHomography(frame, homographyMatrix, newPositions)
        # transformedFrame[450:] = 255
        # transformedFrame[:,300:] = 255
        cv2.imshow('Frame', transformedFrame)
        cv2.waitKey(300)
