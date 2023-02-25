import cv2
import numpy as np

# Load the image
img = cv2.imread('Image/shape1.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Find the moments of the contour
M = cv2.moments(largest_contour)

# Find the center of the contour
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

# Draw a circle at the center of the contour
cv2.circle(img, center, 15, (0, 0, 255), -1)


#Resize picture
width,height = 600,600
imgResize = cv2.resize(img,(width,height))

# Display the image
cv2.imshow('Image', img)
cv2.imshow('Image', imgResize)
cv2.waitKey(0)
cv2.destroyAllWindows()