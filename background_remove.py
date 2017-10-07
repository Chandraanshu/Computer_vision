import cv2
import numpy as np
import video_io

def removeBackground(video):
    """Removes static background from a video.

    Averages out the frames of the video and subtracts this average from each
    frame.

    Args:
        video: Numpy array with shape (numFrames, frameWidth, frameHeight, 3)
            representing the video.

    Returns:
        Video with the static background removed. The return values is a numpy
        array with same shape as the input video.
    """
    average = np.zeros(video.shape[1 : ])

    for frameIdx, frame in enumerate(video):
        average = average * frameIdx / (frameIdx + 1) + frame / (frameIdx + 1)

    # Numpy broadcasting takes care of subtracting average from each frame.
    diff = video - average

    # Note: Negative values need to be set to 0 because converting to uint8
    # turns -2 into 254.
    diff[diff < 0] = 0
    return diff.astype(np.uint8)

if __name__ == '__main__':
    video = video_io.readVideo('traffic.mp4')
    videoBackgroundRemoved = removeBackground(video)
    video_io.displayVideo(videoBackgroundRemoved)
