import numpy as np
import cv2

# Define the input image and filter
z = np.zeros((300, 600), dtype=np.uint8)  # Single-channel black region
v = np.ones((300, 600), dtype=np.uint8) * 255  # Single-channel white region
img = np.vstack((z, v))  # Combine into one image

# Define the edge-detection filter
filter = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])

# Perform convolution using OpenCV
result = cv2.filter2D(img, -1, filter)

# Normalize the result for better visualization
result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

# Display the images
cv2.imshow("Original Image", img)
cv2.imshow("Edge Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
