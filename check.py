#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('kg.jpg')
img = np.array(im)
plt.title('If you see this, you are good to go!')
plt.figure(1).canvas.set_window_title('Env checker')
plt.imshow(img)
plt.show()
