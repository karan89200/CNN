import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# Function to apply filters and detect edges
def apply_edge_detection(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a list of filters
    filters = [
        np.array([[-1, 2, 3], [-4, -5, -16], [-7, 18, 9]]),  # Custom filter 1
        np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),   # Sobel filter X
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),      # Sobel filter Y
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),        # Laplacian filter
        np.array([[-1, -1, -1], [2, 2, 2], [1, 1, 1]]),      # Prewitt filter X
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),      # Prewitt filter Y
        np.array([[-1, -1, 2], [-1, 2, 2], [2, 2, -1]]),     # Edge detection filter 1
        np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]]),        # Edge detection filter 2
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),     # Sobel + High-pass filter
        np.array([[-1, 2, 0], [-1, 2, 0], [0, 0, 0]]),       # Horizontal edge filter
        np.array([[-1, 0, 1], [-1, 0, 1], [0, 0, 0]]),       # Vertical edge filter
        np.array([[1, 1, 1], [1, -6, 1], [1, 1, 1]]),        # Another variation
        np.array([[0, -1, 0], [1, 4, 1], [0, -1, 0]]),       # Simple edge detection
        np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]),       # Simple edge detection X-Y
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])        # Sobel operator for both axes
    ]

    # Apply each filter and store the result
    edge_results = []
    for f in filters:
        edge_result = cv2.filter2D(gray_img, -1, f)
        edge_results.append(edge_result)
    
    return edge_results

# Streamlit web application
def main():
    st.set_page_config(layout="wide")  # Set page layout to wide for better visual
    st.title("Edge Detection with Multiple Filters")

    # Sidebar layout
    with st.sidebar:
        # Image uploader in sidebar
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # If the file is uploaded, process it
    if uploaded_file is not None:
        # Read image
        img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Show original image
        st.image(img, caption="Original Image", use_column_width=True)

        # Apply edge detection
        edge_results = apply_edge_detection(img)

        # Display edge detection results using multiple filters
        st.write("Edge detection results using 15 different filters:")

        # Plot the edge detection results in a grid
        fig, axes = plt.subplots(3, 5, figsize=(15, 10))
        axes = axes.ravel()

        for i in range(15):
            axes[i].imshow(edge_results[i], cmap='gray')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')

        st.pyplot(fig)

if __name__ == "__main__":
    main()
