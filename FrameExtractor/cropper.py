import os
from PIL import Image

def crop_images(directory, crop_bounds):
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # List all files in the directory while preserving their current order
    files = os.listdir(directory)
    print(f"Files found: {files}")

    # Filter for image files that start with 'Id' and are followed by a number
    image_files = [f for f in files if f.startswith('id') and f[2:-4].isdigit()]
    print(f"Image files identified for cropping: {image_files}")

    # Crop each image according to the crop_bounds
    for filename in image_files:
        original_path = os.path.join(directory, filename)
        try:
            with Image.open(original_path) as img:
                # Crop the image
                cropped_img = img.crop(crop_bounds)

                # Save the cropped image
                cropped_img.save(original_path)
                print(f"Cropped and saved '{filename}'")
        except Exception as e:
            print(f"Failed to crop '{filename}': {e}")

# Specify the directory containing the images
image_directory = 'cropped'  # Make sure this path is correct

# Specify the crop boundaries (left, upper, right, lower)
crop_bounds = (600, 300, 1100, 1150)  # Adjust these values based on your needs

# Call the function
crop_images(image_directory, crop_bounds)
