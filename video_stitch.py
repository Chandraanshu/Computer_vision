import cv2
import numpy as np
import video_io
import homography, background_remove


if __name__ == '__main__':
    # video = video_io.readVideo('Person.mp4').astype(np.float32)
    #
    # # diff = video - video[5]
    # # background = cv2.cvtColor(video[5].copy(), cv2.COLOR_BGR2GRAY)
    # background = video[1].copy()
    #
    # for i, frame in enumerate(video):
    #     backgroundRemoved = background_remove.removeBackground(frame, background)
    #     cv2.imshow('Frame', backgroundRemoved.astype(np.uint8))
    #
    #     # Press 'q' to close video.
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break
    #
    # video_io.displayVideo(backgroundRemoved)

    video = video_io.readVideo('Shadow1.mp4').astype(np.float32)
    # background = video[10].copy()

    mask = np.logical_or(video[:, :, :, 1] >= 80, video[:, :, :, 0] >= 80)
    mask = np.logical_or(video[:, :, :, 2] < 120, mask)

    video[mask] = 255

    # cv2.imshow('Frame', video.astype(np.uint8))
    # cv2.waitKey(20000)

    for i, frame in enumerate(video.astype(np.uint8)):
        # backgroundRemoved = background_remove.removeBackground(frame, background)
        # backgroundRemoved[np.any(backgroundRemoved > 120, axis=2)] = 255
        cv2.imshow('Frame', frame)

        # Press 'q' to close video.
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # video_io.displayVideo(video)

    frame = video[0]

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    originalPoints = np.array([
        [100, 10],
        [600, 30],
        [frameHeight - 100, frameWidth - 50],
        [80, frameWidth - 50],
    ])
    finalPoints = np.array([
        [100, 100],
        [400, 100],
        [400, 600],
        [100, 600],
    ])
    homographyMatrix = homography.computeHomography(originalPoints, finalPoints)
    newPoints, oldPoints = homography.computeMapping(frameHeight, frameWidth, homographyMatrix)

    # for point in originalPoints:
    #     drawRectangleOnImage(frame,
    #                          list(point)[::-1],
    #                          6,
    #                          6,
    #                          (0, 0, 255))

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 600,600)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(3000)

    # originalPoints = [list(point) + [1] for point in originalPoints]
    # originalPoints = np.array(originalPoints).transpose()
    # solution = np.matmul(homography, originalPoints)
    # print(solution / solution[-1])

    for frame in video:
        transformedFrame = homography.transformImage(frame, oldPoints, newPoints)
        # transformedFrame[450:] = 255
        # transformedFrame[:,300:] = 255
        cv2.imshow('Frame', transformedFrame)
        cv2.waitKey(1)
