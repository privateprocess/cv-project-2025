import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# display list of images with image channel handling
def display_images(*imgs):
    n = len(imgs)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if len(imgs[i].shape) == 2:  # grayscale
            plt.imshow(imgs[i], cmap='gray')
        else:  # color
            plt.imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()
