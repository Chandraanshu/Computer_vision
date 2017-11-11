import cv2
import numpy as np
import video_io
import homography
import background_remove
import artificial_shadow
import utils
import shadow
import constants


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
    # video = video_io.readVideo(constants.PERSON_VIDEO).astype(np.float32)
    #
    # # diff = video - video[5]
    # # background = cv2.cvtColor(video[5].copy(), cv2.COLOR_BGR2GRAY)
    # background = video[constants.PERSON_BACKGROUND_FRAME].copy()
    #
    # for i, frame in enumerate(video):
    #     backgroundRemoved = background_remove.removeBackground(frame, background)
    #     artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemoved, constants.ARTIFICIAL_SHADOW_OFFSET, constants.ARTIFICIAL_SHADOW_COLOUR)
    #     cv2.imshow('Frame', artificialShadowAdded)
    #
    #     # Press 'q' to close video.
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    print ('start reading')
    shadow_video = video_io.readVideo(constants.SHADOW_VIDEO).astype(np.float32)
    print ('shadow done')
    person_video = video_io.readVideo(constants.PERSON_VIDEO).astype(np.float32)
    print ('person done')
    background = cv2.cvtColor(shadow_video[constants.SHADOW_BACKGROUND_FRAME].copy(), cv2.COLOR_BGR2GRAY)

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

    frameHeight, frameWidth = shadow_video.shape[1], shadow_video.shape[2]

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
    print ('homo done')

    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 600,600)

    for shadow_frame, person_frame in zip(shadow_video, person_video):
        # Work with shadow in grayscale.
        shadow_frame = cv2.cvtColor(shadow_frame, cv2.COLOR_BGR2GRAY)
        backgroundRemoved = background_remove.removeBackground(shadow_frame, background)

        person_frame = np.flip(person_frame, 1).astype(np.uint8)
        person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2GRAY)
        # Threshold out whiter portions.
        backgroundRemoved[backgroundRemoved > constants.SHADOW_THRESHOLD] = 255
        transformedFrame = homography.transformImage(backgroundRemoved, originalPointsCoords, finalPointsCoords)

        # Crop out unneeded portions of image.
        transformedFrame = utils.cropImage(transformedFrame, 150, 230, 50, 300)

        # Expand shadow to become normal sized.
        transformedFrame = utils.imageExpand(transformedFrame, constants.LAPLACIAN_BLUR_WINDOW_SIZE).astype(np.uint8)

        # Find shadow and crop out person
        shadowPosition = shadow.findShadowPosition(transformedFrame, constants.SHADOW_SIZE)
        transformedFrame = utils.cropImage(transformedFrame, 0, 0, 0, transformedFrame.shape[1] - shadowPosition[1] - constants.SHADOW_SIZE[1] - 100)

        #utils.drawTopLeftRectangleOnImage(transformedFrame, shadowPosition[::-1], constants.SHADOW_SIZE[1], constants.SHADOW_SIZE[0], (0, 0, 255))

        shadow_mask = transformedFrame[transformedFrame != 255]

        person_frame[shadow_mask] = transformedFrame[shadow_mask]

        cv2.imshow('Frame', cv2.cvtColor(person_frame, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)
