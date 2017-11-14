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

    shadow_video = video_io.readVideo(constants.SHADOW_VIDEO)
    print("Shadow done")
    person_video = video_io.readVideo(constants.PERSON_VIDEO)
    # background = shadow_video[constants.SHADOW_BACKGROUND_FRAME].copy()
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

    frame = next(shadow_video)

    frameHeight, frameWidth = frame.shape[0], frame.shape[1]

    originalPoints = np.array([
        [320, 800],
        [frameHeight - 380, 800],
        [frameHeight - 370, frameWidth - 550],
        [480, frameWidth - 550],
    ])
    finalPoints = np.array([
        [250, 100],
        [360, 100],
        [360, 485],
        [250, 485],
    ])

    # Uncomment when figuring out points for homography.
    visualizePoints(shadow_video[5], originalPoints)

    homographyMatrix = homography.computeHomography(originalPoints, finalPoints)
    finalPointsCoords, originalPointsCoords = homography.computeMapping(frameHeight, frameWidth, homographyMatrix)
    # print ('homo done')
    personFrameNumber = 0

    for personFrame in person_video:
        if personFrameNumber == constants.PERSON_BACKGROUND_FRAME:
            backgroundPerson = personFrame.copy()

        personFrameNumber += 1

        if personFrameNumber < constants.PERSON_START_FRAME:
            continue
        elif personFrameNumber >= constants.ARTIFICAL_SHADOW_BREAK_FRAME:
            break


        # personFrame = cv2.cvtColor(personFrame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemovedPerson, constants.ARTIFICIAL_SHADOW_OFFSET, constants.ARTIFICIAL_SHADOW_COLOUR)
        artificialShadowAdded = utils.cropImage(artificialShadowAdded, 0, 150, constants.ARTIFICIAL_SHADOW_OFFSET[1], 0)

        cv2.imshow('Frame', artificialShadowAdded)
        cv2.waitKey(1)

    shadowFrameNumber = 0

    for shadowFrame in shadow_video:
        if shadowFrameNumber == constants.SHADOW_BACKGROUND_FRAME:
            background = cv2.cvtColor(shadowFrame.copy().astype(np.uint8), cv2.COLOR_BGR2GRAY)

        if shadowFrameNumber >= constants.SHADOW_START_FRAME:
            break
        shadowFrameNumber += 1

    for shadowFrame, personFrame in zip(shadow_video, person_video):
        # Work with shadow in grayscale.
        shadowFrame = cv2.cvtColor(shadowFrame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        backgroundRemoved = background_remove.removeBackgroundGray(shadowFrame, background)
        # backgroundRemoved = background_remove.removeBackground(shadowFrame, background)

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

        shadowMask = transformedFrame < 250

        transformedFrame = cv2.cvtColor(transformedFrame, cv2.COLOR_GRAY2BGR)

        # personFrame = np.flip(personFrame, 0)
        # personFrame = cv2.cvtColor(personFrame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        # print(backgroundRemovedPerson.shape)
        # backgroundRemovedPerson[:transformedFrame.shape[0], :transformedFrame.shape[1]][shadowMask] = transformedFrame[shadowMask]
        personMask = np.any(backgroundRemovedPerson != 255, axis=2)

        finalFrame = np.full(personFrame.shape, fill_value=255, dtype=np.uint8)
        finalFrame[70:70+transformedFrame.shape[0], :transformedFrame.shape[1]][shadowMask] = transformedFrame[shadowMask]
        finalFrame[personMask] = backgroundRemovedPerson[personMask]

        cv2.imshow('Frame', finalFrame)
        cv2.waitKey(1)

    video_io.shutdown()
