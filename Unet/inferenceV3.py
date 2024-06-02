import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNET  # Ensure you have the correct import for your model
from utils import load_checkpoint  # Ensure you have the correct import for your utils
import matplotlib.pyplot as plt
import time

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained ResUNet model
model = UNET(in_channels=3, out_channels=1).to(device)  # Ensure this matches your model definition
checkpoint = torch.load('my_checkpoint.pth.tar')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()

# Specify input and output directories
input_dir = 'data/test_images'
output_dir = 'data/output_images'
overlay_output_dir = 'data/overlay_images'  # Add a new directory for overlaid images
plot_output_dir = 'plots'  # Directory to save plots
txt_output_path = os.path.join(plot_output_dir, 'width_measurements.txt')

# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(overlay_output_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((850, 500)),  # Resize to match the model's input size
    transforms.ToTensor(),
])

# Create a custom dataset for the test images
class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0].replace('id', '')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load input image
        input_image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)

        return {'image': input_image, 'image_name': image_name, 'image_id': image_name.split('.')[0]}  # Extract ID

dataset = PredictionDataset(input_dir, transform=transform)

# Create a data loader for the test dataset
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize lists to store width measurements and image IDs
width_measurements = []
image_ids = []

# Measure the time taken for predictions
start_time = time.time()

# Iterate through each batch of images in the test dataset
for batch in data_loader:
    # Move the batch to the appropriate device
    images = batch['image'].to(device)
    image_id = batch['image_id'][0]

    # Make predictions using the ResUNet model
    with torch.no_grad():
        pred_mask = model(images)

    # Apply threshold to the predicted mask
    pred_mask = (pred_mask > 0.9).float()

    # Convert the predicted mask to a numpy array
    binary_mask_np = pred_mask.squeeze().cpu().numpy()

    # Convert to uint8
    binary_mask_np_uint8 = (binary_mask_np * 255).astype(np.uint8)

    # Save the result mask to the output directory
    image_name = batch['image_name'][0]
    output_path = os.path.join(output_dir, f'result_mask_{image_name}')
    cv2.imwrite(output_path, binary_mask_np_uint8)

    # Distance Transform to measure width at four static places
    image_widths = []
    if np.any(binary_mask_np_uint8):
        height, width = binary_mask_np_uint8.shape
        positions = [int(height * 0.2), int(height * 0.4), int(height * 0.6), int(height * 0.8)]

        # Resize original image to match binary mask dimensions
        original_image = images.squeeze().cpu().numpy().transpose(1, 2, 0) * 255  # Convert to 0-255 scale
        original_image_resized = cv2.resize(original_image, (binary_mask_np_uint8.shape[1], binary_mask_np_uint8.shape[0])).astype(np.uint8)
        overlay = original_image_resized.copy()  # Copy the resized input image

        for idx, pos in enumerate(positions):
            column = binary_mask_np_uint8[pos, :]
            if np.any(column):
                start = np.argmax(column)
                end = len(column) - 1 - np.argmax(column[::-1])
                width_measurement = end - start
                image_widths.append(width_measurement)
            else:
                image_widths.append(0)

        # Overlay the predicted mask on the input image
        overlay[binary_mask_np_uint8 > 0] = [0, 0, 255]  # Set overlay color where mask is 1 (assuming blue color)
        overlay = cv2.addWeighted(overlay, 0.3, original_image_resized, 0.7, 0)

        # Draw green lines and width measurements on the overlay image
        for idx, pos in enumerate(positions):
            start = np.argmax(binary_mask_np_uint8[pos, :])
            end = len(binary_mask_np_uint8[pos, :]) - 1 - np.argmax(binary_mask_np_uint8[pos, ::-1])
            if start < end:  # Only draw if a valid width is found
                cv2.line(overlay, (start, pos), (end, pos), (0, 255, 0), 3)
                cv2.putText(overlay, f'{image_widths[idx]}', (end + 5, pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        image_widths = [0, 0, 0, 0]

    # Store the width measurements and image IDs in the lists
    width_measurements.append(image_widths)
    image_ids.append(image_id)

    # Save the overlaid image to the overlay output directory
    overlay_output_path = os.path.join(overlay_output_dir, f'overlay_{image_name}')
    cv2.imwrite(overlay_output_path, overlay)

    # Print or save the average width
    print(f"Widths for {image_name}: {image_widths}")

# Calculate and print the total time taken
total_time = time.time() - start_time
print(f"Total time taken for predictions: {total_time:.2f} seconds")

# Save width measurements to a text file
with open(txt_output_path, 'w') as f:
    for image_id, widths in zip(image_ids, width_measurements):
        f.write(f"{image_id}: {widths}\n")

# Plot the width measurements
width_measurements = np.array(width_measurements)

# Plot the width measurements without standardization
plt.figure(figsize=(12, 6))
for i, position in enumerate([0.2, 0.4, 0.6, 0.8]):
    plt.plot(width_measurements[:, i], marker='o', linestyle='-', markersize=2, label=f'Position {position*100:.0f}%')
plt.xlabel('Image ID')
plt.ylabel('Width Measurement')
plt.title('Width Measurements at Four Positions Across Images')
plt.xticks(ticks=np.arange(0, len(image_ids), step=10), labels=image_ids[::10], rotation=90)  # Step size of 10
plt.legend()
plot_path = os.path.join(plot_output_dir, 'width_measurements_plot.png')
plt.savefig(plot_path)
plt.show()
