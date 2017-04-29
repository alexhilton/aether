#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('kg.jpg')
img = np.array(im)
plt.title('If you see this, you are good to go!')
plt.imshow(img)
plt.show()
