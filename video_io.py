import cv2

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


def getAllFrames(cap):
    """Reads all frames from a cv2.VideoCapture object.

    Args:
        cap: A cv2.VideoCapture object.

    Returns:
        Numpy array with shape (numFrames, frameWidth, frameHeight, 3)
        containing all frames in the video.
    """
    video = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
          break

        video.append(frame)

    return video


def shutdown(*caps):
    """Shuts down all cv2.VideoCapture objects passed in and closes all windows.

    Args:
        *caps: Any number of cv2.VideoCapture objects passed in as multiple
            arguments.
    """
    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()

def readVideo(fileName):
    """Reads video from file and returns all frames contained in it.

    Args:
        fileName: The name of the video file.

    Returns:
        Numpy array with shape (numFrames, frameWidth, frameHeight, 3)
        containing all frames in the video.
    """
    cap = openVideo(fileName)
    video = getAllFrames(cap)
    shutdown(cap)

    return video
