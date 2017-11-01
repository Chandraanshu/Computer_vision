from scipy import misc, signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np
import video_io
import utils


if __name__ == '__main__':
    video = video_io.readVideo('traffic.mp4')
    frame = video[0]
    nframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # nframe = misc.imread('orchid_gray.jpg', 'F')
    gx = utils.pixelDiffImages(nframe, 0, 1, nframe, 0, 0, nframe.shape[0] - 1, nframe.shape[1] - 1)
    gy = utils.pixelDiffImages(nframe, 1, 0, nframe, 0, 0, nframe.shape[0] - 1, nframe.shape[1] - 1)

    # print(type(gx))
    # print(type(gy))

    Ixx = gx * gx
    Ixy = gx * gy
    Iyy = gy * gy

    FULLWIN = 13

    gkern = np.outer(signal.gaussian(FULLWIN, 2.5), signal.gaussian(FULLWIN, 2.5))

    Wxx = signal.fftconvolve(Ixx, gkern, mode='same')
    Wxy = signal.fftconvolve(Ixy, gkern, mode='same')
    Wyy = signal.fftconvolve(Iyy, gkern, mode='same')

    W = np.zeros(Wxx.shape)

    for i in range(Wxx.shape[0]):
        if i % 100 == 0:
            print(i)
        for j in range(Wxx.shape[1]):
            W[i][j] = min(np.linalg.eigvals([[Wxx[i][j], Wxy[i][j]], [Wxy[i][j], Wyy[i][j]]]))


    rounded_shape = (np.array(W.shape) // FULLWIN) * FULLWIN

    W = W[ : rounded_shape[0], : rounded_shape[1]]
    for i in range(0, W.shape[0], FULLWIN):
        for j in range(0, W.shape[1], FULLWIN):
            max_val = np.amax(W[i : i + FULLWIN, j : j + FULLWIN])
            for x in range(FULLWIN):
                for y in range(FULLWIN):
                    if W[i + x, j + y] < max_val:
                        W[i + x, j + y] = 0

    # Efficient way of getting indices of 200 largest elements. Note the minus sign in front
    # of W.ravel(), since argpartition 'sorts' in ascending order
    indices = np.argpartition(-W.ravel(), 19)[ : 20]
    row_indices, col_indices = np.unravel_index(indices, W.shape)

    # Convert MxN array into MxNx3 by triplicating each element
    # coloured = np.tile(nframe[..., None], 3)

    currentAxis = plt.gca()

    # Make rectangles on image
    for i, j in zip(row_indices, col_indices):
        currentAxis.add_patch(Rectangle((j, i), 15, 15, facecolor='none', edgecolor='r'))

    # print(coloured.shape)
    plt.imshow(nframe, cmap='gray')
    
    plt.show()
