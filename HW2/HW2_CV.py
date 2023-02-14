import numpy as np
from PIL import Image

# Create a Gaussian filter
def gaussian_filter(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-((x**2 + y**2)/(2*sigma**2)))
    return g / g.sum()

# Create a box filter
def box_filter(size):
    return np.ones((size, size)) / (size ** 2)

# Load an image
im = Image.open("M3d.jpeg")

# Convert the image to a numpy array
im_arr = np.array(im)

# Apply the Gaussian filter to the image
# im_gaussian = np.zeros(im_arr.shape)
# gaussian = gaussian_filter(5, 1)
# for i in range(3):
#     im_gaussian[:,:,i] = np.convolve(np.convolve(im_arr[:,:,i], gaussian, mode='same'), gaussian.T, mode='same')

# Apply the box filter to the image
im_box = np.zeros(im_arr.shape)
box = box_filter(5)
for i in range(3):
    im_box[:,:,i] = np.convolve(np.convolve(im_arr[:,:,i], box, mode='same'), box.T, mode='same')

# Save the filtered images
# im_gaussian = Image.fromarray(im_gaussian.astype('uint8'))
# im_gaussian.save("gaussian_filtered.jpg")
im_box = Image.fromarray(im_box.astype('uint8'))
im_box.save("box_filtered.jpg")
