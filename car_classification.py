# import cv2
# import numpy as np

# # Load YOLO model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# # Load input image
# image = cv2.imread("pic1.jpg")

# # Get image height and width
# (H, W) = image.shape[:2]

# # Convert input image to blob format
# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# # Set input blob for the network
# net.setInput(blob)

# # Run forward pass through the network
# output_layers = net.forward(net.getUnconnectedOutLayersNames())

# # Create an empty list to store bounding boxes
# bounding_boxes = []

# # Loop over the output layers
# for output in output_layers:
#     # Loop over each of the detections
#     for detection in output:
#         # Get the class ID, confidence, and bounding box coordinates
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             box = detection[0:4] * np.array([W, H, W, H])
#             (centerX, centerY, width, height) = box.astype("int")
#             x = int(centerX - (width / 2))
#             y = int(centerY - (height / 2))
#             bounding_boxes.append([x, y, int(width), int(height), confidence, class_id])

# # Apply non-maxima suppression to remove overlapping bounding boxes
# bounding_boxes = cv2.dnn.NMSBoxes(bounding_boxes, 0.5, 0.4)

# # Draw the bounding boxes on the image
# for (startX, startY, endX, endY) in bounding_boxes:
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# # Display the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)


import tensorflow as tf
import matplotlib.pyplot as plt

# Load the image
image = tf.keras.preprocessing.image.load_img('pic8.jpg', target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.keras.applications.mobilenet.preprocess_input(image)

# Load a pre-trained image classification model
model = tf.keras.applications.mobilenet.MobileNet()

# Make predictions on the image
predictions = model.predict(image[tf.newaxis, ...])

# Decode the predictions
predicted_class = tf.keras.applications.mobilenet.decode_predictions(predictions, top=1)[0][0][1]

# Plot the image and print the prediction
print("Predicted class:", predicted_class)

plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
plt.axis('off')
plt.show()
