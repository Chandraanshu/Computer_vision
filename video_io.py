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
    cap.release()

    return video


def displayVideo(frames, FPS=30):
    """Given all frames of a video, displays it at the given FPS.

    Args:
        frames: Numpy array with shape (numFrames, frameWidth, frameHeight, 3)
            containing all frames in the video.
        FPS: Desired frames per second. Default value is 30.
    """
    delay = 1000 // FPS

    for frame in frames:
        cv2.imshow('Frame', frame)
        cv2.waitKey(delay)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = readVideo('traffic.mp4')
    displayVideo(video)
