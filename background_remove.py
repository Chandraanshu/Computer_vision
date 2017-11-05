import cv2
import numpy as np
import video_io

GOOD_THRESHOLD = 40

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
    # Get mask for all pixels which have a large positive difference.
    goodMask = np.any(abs(diff) >= GOOD_THRESHOLD, axis=3)
    # Replace good pixels with original in video.
    diff[goodMask] = video[goodMask]
    # Replace bad pixels with 0.
    diff[np.invert(goodMask)] = 0

    return diff.astype(np.uint8)

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
