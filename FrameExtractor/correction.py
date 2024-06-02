import cv2
import numpy as np
import os

def apply_custom_perspective_transform(directory, transformation_matrix, output_size):
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # List all image files in the directory
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Image files found: {image_files}")

    # Prepare the transformation matrix from provided array
    M = np.array(transformation_matrix, dtype=float).reshape(3, 3)

    # Apply the perspective transformation to each image
    for filename in image_files:
        original_path = os.path.join(directory, filename)
        try:
            original_image = cv2.imread(original_path)
            if original_image is not None:
                # Apply the perspective warp
                warped_image = cv2.warpPerspective(original_image, M, output_size, flags=cv2.INTER_LINEAR)

                # Save the transformed image
                cv2.imwrite(original_path, warped_image)
                print(f"Transformed and saved '{filename}'")
            else:
                print(f"Failed to load '{filename}'")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")

# Specify the directory containing the images
image_directory = 'corrected/corex'  # Update with the correct path

# The transformation matrix as provided
transformation_matrix = [0.11807599135622489, 13.28671335974383, -4081.503160056561,
-7.279086936956649, 1.9589391793076512, 8207.093655838506,
-1.5563790814373272e-05, 0.00032780890673960155, 1.0
]

# Define the output size based on your image dimensions
output_size = (3840, 2160)  # Width x Height in pixels

# Call the function
apply_custom_perspective_transform(image_directory, transformation_matrix, output_size)
