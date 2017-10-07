def removeBackground(video):
    """Removes static background from a video.

    Averages out the frames of the video and subtracts this average from each
    frame.

    Args:
        video: Numpy array with shape (num_frames, frame_width, frame_height, 3)
            representing the video.

    Returns:
        Video with the static background removed. The return values is a numpy
        array with same shape as the input video.
    """
    average = numpy.zeros(video.shape[1 : ])
    
