import cv2
import numpy as np
import video_io
import homography, background_remove, artificial_shadow, utils, shadow


def visualizePoints(frame, points):
    for point in points:
        utils.drawRectangleOnImage(frame,
                                   list(point)[::-1],
                                   6,
                                   6,
                                   (0, 0, 255))

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 600,600)
    cv2.imshow('Frame', frame)
    cv2.waitKey(10000)


if __name__ == '__main__':
    # video = video_io.readVideo('Person.mp4').astype(np.float32)
    #
    # # diff = video - video[5]
    # # background = cv2.cvtColor(video[5].copy(), cv2.COLOR_BGR2GRAY)
    # background = video[1].copy()
    #
    # for i, frame in enumerate(video):
    #     backgroundRemoved = background_remove.removeBackground(frame, background)
    #     artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemoved, [0, 50], [30, 30, 30])
    #     cv2.imshow('Frame', artificialShadowAdded)
    #
    #     # Press 'q' to close video.
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    video = video_io.readVideo('Shadow1.mp4').astype(np.float32)
    background = cv2.cvtColor(video[10].copy(), cv2.COLOR_BGR2GRAY)

    # mask = np.logical_or(video[:, :, :, 1] >= 80, video[:, :, :, 0] >= 80)
    # mask = np.logical_or(video[:, :, :, 2] < 120, mask)
    #
    # video[mask] = 255

    # for i, frame in enumerate(video.astype(np.uint8)):
    #     # backgroundRemoved = background_remove.removeBackground(frame, background)
    #     # backgroundRemoved[np.any(backgroundRemoved > 120, axis=2)] = 255
    #     cv2.imshow('Frame', frame)
    #
    #     # Press 'q' to close video.
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    # video_io.displayVideo(video)

    frameHeight, frameWidth = video.shape[1], video.shape[2]

    originalPoints = np.array([
        [30, 10],
        [frameHeight - 50, 10],
        [frameHeight - 150, frameWidth - 500],
        [250, frameWidth - 500],
    ])
    finalPoints = np.array([
        [250, 100],
        [400, 100],
        [400, 450],
        [250, 450],
    ])

    # Uncomment when figuring out points for homography.
    # visualizePoints(frame, originalPoints)

    homographyMatrix = homography.computeHomography(originalPoints, finalPoints)
    finalPointsCoords, originalPointsCoords = homography.computeMapping(frameHeight, frameWidth, homographyMatrix)

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 600,600)

    for frame in video:
        # Work with shadow in grayscale.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        backgroundRemoved = background_remove.removeBackground(frame, background)

        # Threshold out whiter portions.
        backgroundRemoved[backgroundRemoved > 120] = 255
        transformedFrame = homography.transformImage(backgroundRemoved, originalPointsCoords, finalPointsCoords)

        # Crop out unneeded portions of image.
        transformedFrame = utils.cropImage(transformedFrame, 150, 230, 50, 300)

        # Expand shadow to become normal sized.
        transformedFrame = utils.imageExpand(transformedFrame, 4).astype(np.uint8)

        # Find shadow and crop out person
        trackingBoxSize = [400, 80]
        shadowPosition = shadow.findShadowPosition(transformedFrame, trackingBoxSize)
        transformedFrame = utils.cropImage(transformedFrame, 0, 0, 0, transformedFrame.shape[1] - shadowPosition[1] - trackingBoxSize[1] - 100)

        utils.drawTopLeftRectangleOnImage(transformedFrame, shadowPosition[::-1], trackingBoxSize[1], trackingBoxSize[0], (0, 0, 255))
        cv2.imshow('Frame', transformedFrame)
        cv2.waitKey(1)
