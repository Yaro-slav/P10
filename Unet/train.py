import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 850
IMAGE_WIDTH = 500
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/valid_images"
VAL_MASK_DIR = "data/valid_masks"

def calculate_metrics(predictions, targets, threshold=0.5):
    preds = (predictions > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = intersection / (union + 1e-6)

    true_positives = (preds * targets).sum(dim=(2, 3))
    false_positives = (preds * (1 - targets)).sum(dim=(2, 3))
    false_negatives = ((1 - preds) * targets).sum(dim=(2, 3))

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    dice = (2 * intersection) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6)

    return iou.mean(), precision.mean(), recall.mean(), dice.mean()

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_dice = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # calculate metrics
        iou, precision, recall, dice = calculate_metrics(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), iou=iou.item(), precision=precision.item(), recall=recall.item(), dice=dice.item())
        total_loss += loss.item()
        total_iou += iou.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_dice += dice.item()

    # Return the average loss and metrics for the epoch
    return total_loss / len(loader), total_iou / len(loader), total_precision / len(loader), total_recall / len(loader), total_dice / len(loader)

def validate_fn(loader, model, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)

            predictions = model(x)
            loss = loss_fn(predictions, y)

            # calculate metrics
            iou, precision, recall, dice = calculate_metrics(predictions, y)

            total_loss += loss.item()
            total_iou += iou.item()
            total_precision += precision.item()
            total_recall += recall.item()
            total_dice += dice.item()

    # Return the average loss and metrics for the validation set
    return total_loss / len(loader), total_iou / len(loader), total_precision / len(loader), total_recall / len(loader), total_dice / len(loader)

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # Lists to store losses and metrics for plotting
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_dices = []
    val_dices = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_iou, train_precision, train_recall, train_dice = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_dices.append(train_dice)

        val_loss, val_iou, val_precision, val_recall, val_dice = validate_fn(val_loader, model, loss_fn, device=DEVICE)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_dices.append(val_dice)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}, Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

    # Plot and save the training and validation loss curves
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, NUM_EPOCHS + 1)), train_losses, label='Train Loss')
    plt.plot(list(range(1, NUM_EPOCHS + 1)), val_losses, label='Val Loss')
    plt.xticks(range(1, NUM_EPOCHS + 1))  # Ensure the x-axis shows integers
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('plots/loss_curves.png')

    # Plot and save the training and validation IoU curves
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, NUM_EPOCHS + 1)), train_ious, label='Train IoU')
    plt.plot(list(range(1, NUM_EPOCHS + 1)), val_ious, label='Val IoU')
    plt.xticks(range(1, NUM_EPOCHS + 1))  # Ensure the x-axis shows integers
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU Curves')
    plt.legend()
    plt.savefig('plots/iou_curves.png')

    # Plot and save the training and validation Precision curves
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, NUM_EPOCHS + 1)), train_precisions, label='Train Precision')
    plt.plot(list(range(1, NUM_EPOCHS + 1)), val_precisions, label='Val Precision')
    plt.xticks(range(1, NUM_EPOCHS + 1))  # Ensure the x-axis shows integers
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision Curves')
    plt.legend()
    plt.savefig('plots/precision_curves.png')

    # Plot and save the training and validation Recall curves
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, NUM_EPOCHS + 1)), train_recalls, label='Train Recall')
    plt.plot(list(range(1, NUM_EPOCHS + 1)), val_recalls, label='Val Recall')
    plt.xticks(range(1, NUM_EPOCHS + 1))  # Ensure the x-axis shows integers
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall Curves')
    plt.legend()
    plt.savefig('plots/recall_curves.png')

    # Plot and save the training and validation Dice curves
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, NUM_EPOCHS + 1)), train_dices, label='Train Dice')
    plt.plot(list(range(1, NUM_EPOCHS + 1)), val_dices, label='Val Dice')
    plt.xticks(range(1, NUM_EPOCHS + 1))  # Ensure the x-axis shows integers
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Training and Validation Dice Score Curves')
    plt.legend()
    plt.savefig('plots/dice_curves.png')

if __name__ == "__main__":
    main()
