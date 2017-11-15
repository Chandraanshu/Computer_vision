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

    shadowVideo = video_io.readVideo(constants.SHADOW_VIDEO)
    personVideo = video_io.readVideo(constants.PERSON_VIDEO)
    backgroundImage = cv2.imread('wall.jpg')
    # background = shadowVideo[constants.SHADOW_BACKGROUND_FRAME].copy()

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

    frame = next(shadowVideo)

    frameHeight, frameWidth = frame.shape[0], frame.shape[1]

    originalPoints = np.array([
        [210, 850],
        [frameHeight - 460, 850],
        [frameHeight - 470, frameWidth - 460],
        [380, frameWidth - 470],
    ])
    finalPoints = np.array([
        [250, 100],
        [360, 100],
        [360, 485],
        [250, 485],
    ])

    # Uncomment when figuring out points for homography.
    # visualizePoints(frame, originalPoints)

    homographyMatrix = homography.computeHomography(originalPoints, finalPoints)
    finalPointsCoords, originalPointsCoords = homography.computeMapping(frameHeight, frameWidth, homographyMatrix)
    # print ('homo done')
    personFrameNumber = 0
    first = True

    for personFrame in personVideo:
        if personFrameNumber == constants.PERSON_BACKGROUND_FRAME:
            backgroundPerson = personFrame.copy()

        personFrameNumber += 1

        if personFrameNumber < constants.PERSON_START_FRAME:
            continue
        elif personFrameNumber >= constants.ARTIFICAL_SHADOW_BREAK_FRAME:
            break

        # personFrame = cv2.cvtColor(personFrame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        # artificialShadowAdded = artificial_shadow.addArtificalShadow(backgroundRemovedPerson, constants.ARTIFICIAL_SHADOW_OFFSET, constants.ARTIFICIAL_SHADOW_COLOUR)
        # artificialShadowAdded = utils.cropImage(artificialShadowAdded, 0, 150, constants.ARTIFICIAL_SHADOW_OFFSET[1], 0)

        personPixels = np.any(backgroundRemovedPerson != 255, axis=2)
        personPixelsCropped = personPixels[: backgroundRemovedPerson.shape[0] - constants.ARTIFICIAL_SHADOW_OFFSET[0], : backgroundRemovedPerson.shape[1] - constants.ARTIFICIAL_SHADOW_OFFSET[1]]

        personMask = np.any(backgroundRemovedPerson[:, constants.PERSON_MOVE:] != 255, axis=2)
        finalFrame = backgroundImage.copy()[:backgroundRemovedPerson.shape[0], :backgroundRemovedPerson.shape[1]]

        finalFrame[constants.ARTIFICIAL_SHADOW_OFFSET[0] : , constants.ARTIFICIAL_SHADOW_OFFSET[1] : ][personPixelsCropped] -= constants.ARTIFICIAL_SHADOW_COLOUR
        # finalFrame = np.full(artificialShadowAdded.shape, fill_value=255, dtype=np.uint8)
        finalFrame[:, :-constants.PERSON_MOVE][personMask] = backgroundRemovedPerson[:, constants.PERSON_MOVE:][personMask]
        # finalFrame[:, :-constants.PERSON_MOVE][personMask] -= constants.ARTIFICIAL_SHADOW_COLOUR

        finalFrame = utils.cropImage(finalFrame, 0, 150, 0, constants.PERSON_MOVE)

        if first:
            video_io.writeVideo(finalFrame, 'final1.mp4')
            first = False

        video_io.write(finalFrame)

        cv2.imshow('Frame', finalFrame)
        cv2.waitKey(1)

    for personFrame in personVideo:
        # personFrame = np.flip(personFrame, 0)
        # personFrame = cv2.cvtColor(personFrame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        # print(backgroundRemovedPerson.shape)
        # backgroundRemovedPerson[:transformedFrame.shape[0], :transformedFrame.shape[1]][shadowMask] = transformedFrame[shadowMask]
        personMask = np.any(backgroundRemovedPerson[:, constants.PERSON_MOVE:] != 255, axis=2)

        finalFrame = backgroundImage.copy()[:personFrame.shape[0], :personFrame.shape[1]]
        # finalFrame = np.full(personFrame.shape, fill_value=255, dtype=np.uint8)
        finalFrame[:, :-constants.PERSON_MOVE][personMask] = backgroundRemovedPerson[:, constants.PERSON_MOVE:][personMask]

        finalFrame = utils.cropImage(finalFrame, 0, 150, 0, constants.PERSON_MOVE)

        video_io.write(finalFrame)

        cv2.imshow('Frame', finalFrame)
        cv2.waitKey(1)

        personFrameNumber += 1

        if personFrameNumber >= constants.SHADOW_ENTRY_FRAME:
            break

    shadowFrameNumber = 0

    for shadowFrame in shadowVideo:
        if shadowFrameNumber == constants.SHADOW_BACKGROUND_FRAME:
            background = cv2.cvtColor(shadowFrame.copy().astype(np.uint8), cv2.COLOR_BGR2GRAY)

        if shadowFrameNumber >= constants.SHADOW_START_FRAME:
            break
        shadowFrameNumber += 1

    for shadowFrame, personFrame in zip(shadowVideo, personVideo):
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
        transformedFrame[:, shadowPosition[1] + constants.SHADOW_SIZE[1] + 130 : ] = 255

        # utils.drawTopLeftRectangleOnImage(transformedFrame, shadowPosition[::-1], constants.SHADOW_SIZE[1], constants.SHADOW_SIZE[0], (0, 0, 255))
        transformedFrame = np.flip(transformedFrame, 1)

        shadowMask = transformedFrame < 250

        transformedFrame = cv2.cvtColor(transformedFrame, cv2.COLOR_GRAY2BGR)

        # personFrame = np.flip(personFrame, 0)
        # personFrame = cv2.cvtColor(personFrame, cv2.COLOR_BGR2GRAY)
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        # print(backgroundRemovedPerson.shape)
        # backgroundRemovedPerson[:transformedFrame.shape[0], :transformedFrame.shape[1]][shadowMask] = transformedFrame[shadowMask]
        personMask = np.any(backgroundRemovedPerson[:, constants.PERSON_MOVE:] != 255, axis=2)

        finalFrame = backgroundImage.copy()[:personFrame.shape[0], :personFrame.shape[1]]
        # finalFrame = np.full(personFrame.shape, fill_value=255, dtype=np.uint8)
        finalFrame[70 : 70 + transformedFrame.shape[0], :transformedFrame.shape[1]][shadowMask] -= constants.ARTIFICIAL_SHADOW_COLOUR
        finalFrame[:, :-constants.PERSON_MOVE][personMask] = backgroundRemovedPerson[:, constants.PERSON_MOVE:][personMask]

        finalFrame = utils.cropImage(finalFrame, 0, 150, 0, constants.PERSON_MOVE)

        video_io.write(finalFrame)

        cv2.imshow('Frame', finalFrame)
        cv2.waitKey(1)

    video_io.shutdown()
