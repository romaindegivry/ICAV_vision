### Test ImageIO ImRead and Video Feed

import imageio
import numpy as np

img = imageio.imread('5_far.jpg', format = 'jpg')

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

reader = imageio.get_reader("<video0>")

length = reader.get_length()

print(length)


while True:
    img = reader.get_next_data()
    plt.imshow(np.fliplr(img))
    plt.pause(1/30)

reader.close()