from ultralytics import YOLO
import cv2
import os
import numpy as np

# Paths
model_path = 'runs/segment/train/weights/best.pt'
input_folder = 'data/test/'
output_folder = 'data/output/'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the model
model = YOLO(model_path)

# Alpha value for blending
alpha = 0.5

# Threshold value
threshold = 0.99

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        H, W, _ = img.shape

        results = model(img)

        for i, result in enumerate(results):
            for j, mask in enumerate(result.masks.data):
                mask = mask.cpu().numpy()  # Move tensor to CPU and convert to NumPy

                # Apply the threshold
                mask = (mask > threshold).astype(np.uint8) * 255  # Apply threshold and scale to [0, 255]

                mask = cv2.resize(mask, (W, H))

                # Convert mask to 3-channel image
                mask_3ch = np.stack([mask]*3, axis=-1)

                # Overlay mask on original image
                overlay = cv2.addWeighted(img, 1 - alpha, mask_3ch, alpha, 0)

                output_path = os.path.join(output_folder, f'{os.path.splitext(image_name)[0]}_overlay_{j}.png')
                cv2.imwrite(output_path, overlay)
