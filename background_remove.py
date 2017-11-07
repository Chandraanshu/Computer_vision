import cv2
import numpy as np
import video_io

GOOD_THRESHOLD = 60


def removeBackground(frame, background):
    """Removes static background from a frame.

    Args:
        frame: Numpy array with shape (numFrames, frameWidth, frameHeight, 3)
            representing the frame.

    Returns:
        Video with the static background removed. The return values is a numpy
        array with same shape as the input frame.
    """
    # Numpy broadcasting takes care of subtracting average from each frame.
    # frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Note: Negative values need to be set to 0 because converting to uint8
    # turns -2 into 254.
    # Get mask for all pixels which have a large positive difference.
    badMask = np.all(abs(frame - background) < GOOD_THRESHOLD, axis=2)
    # Replace good pixels with original in frame.
    frame[badMask] = 255
    return frame


def getVideoAverage(video):
    average = np.zeros(video.shape[1 : ])

    for frameIdx, frame in enumerate(video):
        average = average * frameIdx / (frameIdx + 1) + frame / (frameIdx + 1)

    return average


if __name__ == '__main__':
    video = (video_io.readVideo('WALL.MOV')[0:250]).astype(np.float32)
    #videoBackgroundRemoved = removeBackground(video)
    first_frame = video[1].copy()

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 600,600)
    for frame in video:
        new_video = np.abs(frame - first_frame)
        goodMask = np.any(new_video >= 40, axis=2)
        frame[np.invert(goodMask)] = 255
        cv2.imshow('Frame', frame.astype(np.uint8))
        cv2.waitKey(50)
    # video_io.displayVideo(video.astype(np.uint8))
    #video_io.writeVideo(videoBackgroundRemoved, 'bg_removed.mp4')
