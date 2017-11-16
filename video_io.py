import os
import time
import cv2
import numpy as np


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
        Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
        containing all frames in the video.
    """
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        yield frame.astype(np.float32)

    cap.release()


def readVideo(fileName, numFrames=0):
    """Reads video from file and returns all frames contained in it.

    Args:
        fileName: The name of the video file.

    Returns:
        Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
        containing all frames in the video.
    """
    cap = openVideo(fileName)
    return getAllFrames(cap)


def shutdown():
    WRITER.release()


def displayVideo(frames, FPS=30):
    """Given all frames of a video, displays it at the given FPS.

    Args:
        frames: Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
            containing all frames in the video.
        FPS: Desired frames per second. Default value is 30.
    """
    delay = 1000 // FPS

    for frame in frames:
        cv2.imshow('Frame', frame)

        # Press 'q' to close video.
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def createVideoWriter(frame, fileName, FPS=30):
    """Given a video, write it to a file.

    Deletes the specified file, if already present.

    Args:
        video: Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
            containing all frames in the video.
        fileName: Name of file to which the video should be written.
        FPS: Desired frames per second. Default value is 30.
    """
    try:
        os.remove(fileName)
    except FileNotFoundError:
        pass

    return cv2.VideoWriter(
        fileName,
        apiPreference=cv2.CAP_ANY,
        fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
        fps=FPS,
        frameSize=tuple(reversed(frame.shape[0 : 2]))
    )


def writeFrame(writer, frame):
    writer.write(frame)


def displayFrame(frame, waitTime=1):
    cv2.imshow('Frame', frame)
    cv2.waitKey(waitTime)


if __name__ == '__main__':
    video = readVideo('traffic.mp4')
    displayVideo(video)
    writeVideo(video, 'copy.mp4')
