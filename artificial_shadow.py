import numpy as np


def addArtificalShadow(frame, shadowOffset, shadowColor):
    """Adds an artifical shadow to a person in a frame.

    Every pixel except for those that are part of the person must be completely
    white.
    The shadow is placed 'behind' the person.
    The shadow is created by simply shifting the person by the offset and
    colouring all the pixels a dark colour.

    Args:
        frame: A numpy array containing the frame.
        shadowOffset: A list with 2 numbers, representing how much the shadow is
            shifted from the original person, in matrix coordinates.
        shadowColor: Gives the colour of the shadow. Can be in one of two
            formats:
            1) A list with 3 numbers, giving a BGR colour.
            2) A single number, in which case all of B, G and R will be given
               the same value.

    Returns:
        A new picture with a shadow added to the person. Pixels not part of the
        person or the shadow will be coloured white.
    """
    # Mask the pixels inside the person
    personPixels = np.any(frame != 255, axis=2)
    personPixelsCropped = personPixels[: frame.shape[0] - shadowOffset[0], : frame.shape[1] - shadowOffset[1]]

    newFrame = np.full(frame.shape, 255, dtype=np.uint8)

    # Add shadow to new frame
    newFrame[shadowOffset[0] : , shadowOffset[1] : ][personPixelsCropped] = shadowColor

    # Add person to new frame
    newFrame[personPixels] = frame[personPixels]

    return newFrame
