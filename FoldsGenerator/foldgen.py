import os
import shutil
import random

# Set the paths to your images and labels folders
images_folder = "images"
labels_folder = "labels"

# Define the number of folds
num_folds = 3

# Get list of images and labels
images = sorted([f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))])
labels = sorted([f for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))])

# Ensure images and labels match
assert len(images) == len(labels), "Number of images and labels must be the same."
for img, lbl in zip(images, labels):
    assert os.path.splitext(img)[0] == os.path.splitext(lbl)[0], f"Image and label file names must match: {img} vs {lbl}"

# Shuffle data
data = list(zip(images, labels))
random.shuffle(data)

# Split data into folds
folds = [[] for _ in range(num_folds)]
for idx, item in enumerate(data):
    folds[idx % num_folds].append(item)

# Create folds directories and distribute data
for fold_idx in range(num_folds):
    fold_folder = f"fold{fold_idx + 1}"

    # Create fold directory
    os.makedirs(fold_folder, exist_ok=True)

    # Create subdirectories for train, val, and test
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(fold_folder, f"{subset}_images"), exist_ok=True)
        os.makedirs(os.path.join(fold_folder, f"{subset}_labels"), exist_ok=True)

    # Assign data to train, val, and test sets
    for idx in range(num_folds):
        if idx == fold_idx:
            subset = "test"
        elif (idx + 1) % num_folds == fold_idx:
            subset = "val"
        else:
            subset = "train"

        for img, lbl in folds[idx]:
            shutil.copy(os.path.join(images_folder, img), os.path.join(fold_folder, f"{subset}_images", img))
            shutil.copy(os.path.join(labels_folder, lbl), os.path.join(fold_folder, f"{subset}_labels", lbl))

print("3-fold cross-validation folders created successfully.")
