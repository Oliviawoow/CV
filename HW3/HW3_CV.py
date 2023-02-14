import cv2
import numpy as np

# Load the image
image = cv2.imread("face8.jpg")

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of skin color in HSV
lower_skin = np.array([0, 60, 100], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Threshold the image based on the skin color range
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Remove the smallest, noisy connected components
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Dilate and erode the remaining blobs
dilation = cv2.dilate(opening,kernel,iterations = 2)
erosion = cv2.erode(dilation,kernel,iterations = 2)

# Find the contours in the image
contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours
for cnt in contours:
    # Check the number of points in the contour
    if len(cnt) >= 5:
        # Compute the region properties of the contour
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        # Check if the region is sufficiently round
        if MA/ma > 0.5:
            # Draw the ellipse on the image
            cv2.ellipse(image, (int(x),int(y)),(int(MA/2),int(ma/2)),angle,0,360,(255,0,0),5)

# Show the image with the ellipses drawn on it
cv2.imwrite("face_detect.jpg", image)
