import numpy as np

# background_remove
PERSON_BACKGROUND_GOOD_THRESHOLD = 50
SHADOW_BACKGROUND_GOOD_THRESHOLD = 45

# lk_tracker
TRACK_WINDOW_SIZE = 31  # Must be an odd number
GAUSS_BLUR_WINDOW_SIZE = 17
PYRAMID_DEPTH = 3
PIXEL_TO_TRACK = np.array([109, 300])
NUM_FRAMES_TO_TRACK = 200
# GAUSS_STDDEV = 2.5

# utils
GAUSS_STDDEV = 2.5

# video_stitch
PERSON_VIDEO = 'PerS4.MOV'
PERSON_START_FRAME = 170
PERSON_BACKGROUND_FRAME = 5
ARTIFICIAL_SHADOW_OFFSET = 60
ARTIFICIAL_SHADOW_COLOUR_DEPRESS = 70
ARTIFICAL_SHADOW_BREAK_FRAME = 767
SHADOW_ENTRY_FRAME = 777
PERSON_MOVE = 115

SHADOW_VIDEO = 'ShadS1.mp4'
SHADOW_START_FRAME = 11
SHADOW_BACKGROUND_FRAME = 10
SHADOW_THRESHOLD = 150
LAPLACIAN_BLUR_WINDOW_SIZE = 4
SHADOW_SIZE = [400, 80]
SHADOW_PERSON_DISTANCE = 130
SHADOW_VERTICAL_POSITION = 70

BACKGROUND_IMAGE = 'wall.jpg'

OUTPUT_VIDEO = 'final1.mp4'
