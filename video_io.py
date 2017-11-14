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


def getAllFrames(cap, numFrames):
    """Reads all frames from a cv2.VideoCapture object.

    Args:
        cap: A cv2.VideoCapture object.

    Returns:
        Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
        containing all frames in the video.
    """
    video = []
    num = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        video.append(frame)
        num += 1
        if num == numFrames:
            break

        yield frame.astype(np.float32)

    # return np.array(video)

CAP = {}


def readVideo(fileName, numFrames=0):
    """Reads video from file and returns all frames contained in it.

    Args:
        fileName: The name of the video file.

    Returns:
        Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
        containing all frames in the video.
    """
    CAP[fileName] = openVideo(fileName)
    return getAllFrames(CAP[fileName], numFrames)

def shutdown():
    for cap in CAP.values():
        cap.release()
    # return video


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


def writeVideo(video, fileName, FPS=30):
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

    outWriter = cv2.VideoWriter(
        fileName,
        apiPreference=cv2.CAP_ANY,
        fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
        fps=FPS,
        frameSize=tuple(reversed(video.shape[1 : 3]))
    )

    for frame in video:
        outWriter.write(frame)
        time.sleep(0.01)  # Needs some time to write the frame

    outWriter.release()


if __name__ == '__main__':
    video = readVideo('traffic.mp4')
    displayVideo(video)
    writeVideo(video, 'copy.mp4')
