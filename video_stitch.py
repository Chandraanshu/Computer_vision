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

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 800)
    cv2.imshow('Frame', frame.astype(np.uint8))
    cv2.waitKey(10000)


if __name__ == '__main__':
    # video = video_io.readVideo(constants.PERSON_VIDEO).astype(np.float32)
    #
    # # diff = video - video[5]
    # # background = cv2.cvtColor(video[5].copy(), cv2.COLOR_BGR2GRAY)

    #
    # for i, frame in enumerate(video):
    #     backgroundRemoved = background_remove.removeBackground(frame, background)
    #     artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemoved, constants.ARTIFICIAL_SHADOW_OFFSET, constants.ARTIFICIAL_SHADOW_COLOUR)
    #     cv2.imshow('Frame', artificialShadowAdded)
    #
    #     # Press 'q' to close video.
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    shadow_video = video_io.readVideo(constants.SHADOW_VIDEO, 400).astype(np.float32)
    print("Shadow done")
    person_video = video_io.readVideo(constants.PERSON_VIDEO, 400).astype(np.float32)
    background = cv2.cvtColor(shadow_video[constants.SHADOW_BACKGROUND_FRAME].copy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # background = shadow_video[constants.SHADOW_BACKGROUND_FRAME].copy()
    backgroundPerson = person_video[constants.PERSON_BACKGROUND_FRAME].copy()
    backgroundPerson = np.flip(backgroundPerson, 0)
    print("done reading")

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
        [260, 620],
        [frameHeight - 440, 620],
        [frameHeight - 470, frameWidth - 740],
        [400, frameWidth - 740],
    ])
    finalPoints = np.array([
        [250, 100],
        [360, 100],
        [360, 485],
        [250, 485],
    ])

    # Uncomment when figuring out points for homography.
    # visualizePoints(shadow_video[5], originalPoints)

    homographyMatrix = homography.computeHomography(originalPoints, finalPoints)
    finalPointsCoords, originalPointsCoords = homography.computeMapping(frameHeight, frameWidth, homographyMatrix)
    # print ('homo done')

    for person_frame in person_video[constants.PERSON_START_FRAME : constants.ARTIFICAL_SHADOW_BREAK_FRAME]:



    for shadow_frame, person_frame in zip(shadow_video, person_video):
        # Work with shadow in grayscale.
        shadow_frame = cv2.cvtColor(shadow_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        backgroundRemoved = background_remove.removeBackgroundGray(shadow_frame, background)
        # backgroundRemoved = background_remove.removeBackground(shadow_frame, background)

        # Threshold out whiter portions.
        backgroundRemoved[backgroundRemoved > constants.SHADOW_THRESHOLD] = 255
        transformedFrame = homography.transformImage(backgroundRemoved, originalPointsCoords, finalPointsCoords)

        # transformedFrame = cv2.cvtColor(transformedFrame, cv2.COLOR_BGR2GRAY)

        # Crop out unneeded portions of image.
        transformedFrame = utils.cropImage(transformedFrame, 200, 600, 0, 1400)

        # Expand shadow to become normal sized.
        transformedFrame = utils.imageExpand(transformedFrame, constants.LAPLACIAN_BLUR_WINDOW_SIZE).astype(np.uint8)

        # Find shadow and crop out person
        shadowPosition = shadow.findShadowPosition(transformedFrame, constants.SHADOW_SIZE)
        transformedFrame = utils.cropImage(transformedFrame, 0, 0, 0, 200)
        transformedFrame = utils.cropImage(transformedFrame, 2, 2, 2, 2)
        transformedFrame[:, shadowPosition[1] + constants.SHADOW_SIZE[1] + 100 : ] = 255

        utils.drawTopLeftRectangleOnImage(transformedFrame, shadowPosition[::-1], constants.SHADOW_SIZE[1], constants.SHADOW_SIZE[0], (0, 0, 255))
        transformedFrame = np.flip(transformedFrame, 1)

        shadow_mask = transformedFrame < 250

        transformedFrame = cv2.cvtColor(transformedFrame, cv2.COLOR_GRAY2BGR)

        person_frame = np.flip(person_frame, 0)
        # person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(person_frame, backgroundPerson)
        # print(backgroundRemovedPerson.shape)
        artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemovedPerson, constants.ARTIFICIAL_SHADOW_OFFSET, constants.ARTIFICIAL_SHADOW_COLOUR)
        artificialShadowAdded = utils.cropImage(artificialShadowAdded, 0, 150, constants.ARTIFICIAL_SHADOW_OFFSET[1], 0)

        artificialShadowAdded[:transformedFrame.shape[0], :transformedFrame.shape[1]][shadow_mask] = transformedFrame[shadow_mask]

        cv2.imshow('Frame', artificialShadowAdded)
        cv2.waitKey(1)
