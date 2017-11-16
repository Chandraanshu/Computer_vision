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


def addShadowToBackground(background, shadow):
    shadowMask = shadow < 250
    background[constants.SHADOW_VERTICAL_POSITION : constants.SHADOW_VERTICAL_POSITION + shadow.shape[0], :shadow.shape[1]][shadowMask] -= constants.ARTIFICIAL_SHADOW_COLOUR
    return background


if __name__ == '__main__':
    shadowVideo = video_io.readVideo(constants.SHADOW_VIDEO)
    personVideo = video_io.readVideo(constants.PERSON_VIDEO)
    backgroundImage = cv2.imread(constants.BACKGROUND_IMAGE)


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
            # Stop adding artificial shadow at this time.
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
        video_io.displayFrame(finalFrame)


    # Create video of person with no shadow.
    for personFrame in personVideo:
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)

        finalFrame = backgroundImage.copy()[:backgroundRemovedPerson.shape[0], :backgroundRemovedPerson.shape[1]]

        finalFrame = addPersonToBackground(finalFrame, backgroundRemovedPerson)

        video_io.write(finalFrame)
        video_io.displayFrame(finalFrame)

        personFrameNumber += 1

        if personFrameNumber >= constants.SHADOW_ENTRY_FRAME:
            # Need to start adding the "real" shadow.
            break


    # Run through extra frames at the beginning of the shadow video.
    shadowFrameNumber = 0

    for shadowFrame in shadowVideo:
        if shadowFrameNumber == constants.SHADOW_BACKGROUND_FRAME:
            # Pull in backgroundFrame in grayscale.
            backgroundShadow = cv2.cvtColor(shadowFrame.copy().astype(np.uint8), cv2.COLOR_BGR2GRAY)

        if shadowFrameNumber >= constants.SHADOW_START_FRAME:
            # Need to start adding shadow to video.
            break

        shadowFrameNumber += 1


    # Create video with "real" shadow.
    for shadowFrame, personFrame in zip(shadowVideo, personVideo):
        # Work with shadow in grayscale.
        shadowFrame = cv2.cvtColor(shadowFrame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        backgroundRemovedShadow = background_remove.removeBackgroundGray(shadowFrame, backgroundShadow)

        # Threshold out whiter portions.
        backgroundRemovedShadow[backgroundRemovedShadow > constants.SHADOW_THRESHOLD] = 255
        transformedShadowFrame = homography.transformImage(backgroundRemovedShadow, originalPointsCoords, finalPointsCoords)

        # Crop out unneeded portions of image.
        transformedShadowFrame = utils.cropImage(transformedShadowFrame, 200, 600, 0, 1400)

        # Expand shadow to become normal sized.
        transformedShadowFrame = utils.imageExpand(transformedShadowFrame, constants.LAPLACIAN_BLUR_WINDOW_SIZE).astype(np.uint8)

        # Find shadow and whiten out person
        shadowPosition = shadow.findShadowPosition(transformedShadowFrame, constants.SHADOW_SIZE)
        transformedShadowFrame = utils.cropImage(transformedShadowFrame, 2, 2, 2, 202)
        transformedShadowFrame[:, shadowPosition[1] + constants.SHADOW_SIZE[1] + constants.SHADOW_PERSON_DISTANCE : ] = 255

        # Horizontally flip shadow video.
        transformedShadowFrame = np.flip(transformedShadowFrame, 1)

        finalFrame = backgroundImage.copy()[:personFrame.shape[0], :personFrame.shape[1]]

        # Add shadow to the background.
        finalFrame = addShadowToBackground(finalFrame, transformedShadowFrame)

        # Add person to background.
        backgroundRemovedPerson = background_remove.removeBackground(personFrame, backgroundPerson)
        finalFrame = addPersonToBackground(finalFrame, backgroundRemovedPerson)

        video_io.write(finalFrame)
        video_io.displayFrame(finalFrame)

    video_io.shutdown()
