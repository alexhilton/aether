#!/usr/bin/env python3

from PIL import Image
import numpy as np
from scipy.ndimage import filters
from matplotlib import pyplot as plt


fig = plt.figure(1)
fig.canvas.set_window_title('Gaussian blur example')
im = np.array(Image.open('kg.jpg'))
origin = fig.add_subplot(1, 4, 1)
origin.imshow(im)
origin.set_title('Original')
blur = fig.add_subplot(1, 4, 2)
im2 = filters.gaussian_filter(im, 2)
blur.imshow(im2)
blur.set_title('Blur with 2')

b3 = fig.add_subplot(1, 4, 3)
im3 = filters.gaussian_filter(im, 3)
b3.imshow(im3)
b3.set_title('Blur with 3')

b5 = fig.add_subplot(1, 4, 4)
im5 = filters.gaussian_filter(im, 5)
b5.imshow(im5)
b5.set_title('Blur with 5')

plt.show()