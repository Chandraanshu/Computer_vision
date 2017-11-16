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


def addArtificialShadowToBackground(background, person):
    personMask = np.any(person[:, constants.PERSON_MOVE - constants.ARTIFICIAL_SHADOW_OFFSET:] != 255, axis=2)
    background[:, : -constants.PERSON_MOVE + constants.ARTIFICIAL_SHADOW_OFFSET][personMask] -= constants.ARTIFICIAL_SHADOW_COLOUR
    return background


def addPersonToBackground(background, person):
    personMask = np.any(person[:, constants.PERSON_MOVE:] != 255, axis=2)
    background[:, : -constants.PERSON_MOVE][personMask] = person[:, constants.PERSON_MOVE :][personMask]
    background = utils.cropImage(background, 0, 150, 0, constants.PERSON_MOVE)
    return background


if __name__ == '__main__':
    shadowVideo = video_io.readVideo(constants.SHADOW_VIDEO)
    personVideo = video_io.readVideo(constants.PERSON_VIDEO)
    backgroundImage = cv2.imread('wall.jpg')


    # Create homography
    shadowFrame = next(shadowVideo)

    frameHeight, frameWidth = shadowFrame.shape[0], shadowFrame.shape[1]

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


    # Create video of person with artificial shadow.
    personFrameNumber = 0
    firstFrame = True

    for personFrame in personVideo:
        # Pull in background frame.
        if personFrameNumber == constants.PERSON_BACKGROUND_FRAME:
            backgroundPerson = personFrame.copy()

        personFrameNumber += 1

        if personFrameNumber < constants.PERSON_START_FRAME:
            # Skip frames before start.
            continue
        elif personFrameNumber >= constants.ARTIFICAL_SHADOW_BREAK_FRAME:
            # Stop adding artificial shadow at this time
            break

        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)

        finalFrame = backgroundImage.copy()[:backgroundRemovedPerson.shape[0], :backgroundRemovedPerson.shape[1]]

        finalFrame = addArtificialShadowToBackground(finalFrame, backgroundRemovedPerson)
        finalFrame = addPersonToBackground(finalFrame, backgroundRemovedPerson)

        if firstFrame:
            # Needs size of video to create the video writer.
            video_io.writeVideo(finalFrame, 'final1.mp4')
            firstFrame = False

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
