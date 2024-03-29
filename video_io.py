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
    """Reads a given number of frames from a cv2.VideoCapture object.

    Args:
        cap: A cv2.VideoCapture object.
        numFrames: Specifies the number of frames to be read in. If 0, then all
        the frames will be read.

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

    return np.array(video)


def getAllFramesAsGenerator(cap):
    """Yields the frames in a video one by one.

    Args:
        cap: A cv2.VideoCapture object.

    Yields:
        Each frame of the video.
    """
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        yield frame.astype(np.float32)

    cap.release()


def readVideo(fileName, numFrames=0):
    """Reads video from file and returns a given number of the frames contained in it.

    Args:
        fileName: The name of the video file.
        numFrames: A default argument specifying the number of frames of the
            video to be read in. Defaults to 0, which reads in all frames.

    Returns:
        Numpy array with shape (numFrames, frameHeight, frameWidth, 3)
        containing all frames in the video.
    """
    cap = openVideo(fileName)
    video = getAllFrames(cap, numFrames)
    cap.release()

    return video


def readVideoAsGenerator(fileName):
    """Reads in a video as a generator that can be iterated over.

    Args:
        fileName: Name of the file from which the video shall be read.

    Yields:
        Each frame of the video.
    """
    cap = openVideo(fileName)
    return getAllFramesAsGenerator(cap)


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
    """Creates a video writer that can write frames with same size as the given one to file.

    Deletes the specified file, if already present.

    Args:
        frame: Numpy array containing a frame. Used to specify the size of the
            frames in the video.
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
        fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
        fps=FPS,
        frameSize=tuple(reversed(frame.shape[0 : 2]))
    )


def writeFrame(writer, frame):
    """Writes a frame to file using the given video writer.

    Args:
        writer: The cv2 VideoWriter to use.
        frame: The frame to write to file.
    """
    writer.write(frame)


def displayFrame(frame, waitTime=1):
    """Displays a frame and waits for the given period.

    Args:
        frame: The frame to display.
        waitTime: An optional argument giving the amount of time to wait for.
            Defaults to 1ms.
    """
    cv2.imshow('Frame', frame)
    cv2.waitKey(waitTime)


if __name__ == '__main__':
    video = readVideo('traffic.mp4')
    displayVideo(video)
    writeVideo(video, 'copy.mp4')
