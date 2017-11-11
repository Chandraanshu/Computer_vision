import numpy as np


def findShadowPosition(frame, trackingBoxSize):
    """Gets the position of a shadow in a frame.

    Frame must be in grayscale.
    Finds shadow by assuming that it is the darkest portion of the image.
    Specifically, it finds the rectangle with given size that has the minimum
    sum.

    Args:
        frame: Frame in which to find shadow.
        trackingBoxSize: Approximate size of rectangle that fits around shadow.

    Returns:
        A numpy array containing the matrix coordinates of the top left corner
        of the box thought to contain the shadow.
    """
    # Size of box for which minimum sum will be found.
    prefixSums = np.cumsum(np.cumsum(frame, axis=0), axis=1)

    # Array containing sums of (almost) all subarrays with stipulated size.
    subArraySums = (prefixSums[trackingBoxSize[0] :, trackingBoxSize[1] :] -
                    prefixSums[trackingBoxSize[0] :, : prefixSums.shape[1] - trackingBoxSize[1]] -
                    prefixSums[: prefixSums.shape[0] - trackingBoxSize[0], trackingBoxSize[1] :] +
                    prefixSums[: prefixSums.shape[0] - trackingBoxSize[0], : prefixSums.shape[1] - trackingBoxSize[1]])

    # Get index of minimum value. Add 1 to both coordinates due to border issues.
    return np.array(np.unravel_index(np.argmin(subArraySums), subArraySums.shape)) + 1
