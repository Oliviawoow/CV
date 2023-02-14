import cv2
import numpy as np

def detect_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Perform Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Draw the lines on the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the image with the lines drawn on it
    cv2.imwrite("lines_detected.jpg", image)

def detect_rectangles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the rectangles
    rect_list = []
    for contour in contours:
        # Approximate the contour to a polygon
        polygon = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)

        # Check if the polygon is a rectangle
        if len(polygon) == 4:
            x, y, w, h = cv2.boundingRect(polygon)
            rect = (x, y, x + w, y + h)

            # Check if the rectangle has a sufficient area and aspect ratio
            if w * h < 100 or w / h > 2 or h / w > 2:
                continue

            rect_list.append(rect)

    # Draw the rectangles
    for rect in rect_list:
        x1, y1, x2, y2 = rect
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    # Save the image with the rectangles drawn on it
    cv2.imwrite("rectangles_detected.jpg", image)

detect_lines("chess.png")
image_lines = "lines_detected.jpg"
detect_rectangles(image_lines)
