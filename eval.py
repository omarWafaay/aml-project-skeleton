import os
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb

from utils.load_transformed import custom_transformer
from utils.train_models import SimpleTinyImageNetNet


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    print(f"\n📊 Evaluation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    wandb.log({"Eval Loss": avg_loss, "Eval Accuracy": accuracy})
    return accuracy


def main():
    # Initialize wandb for evaluation
    wandb.init(project="tiny-imagenet", name="eval-run")

    # Load validation data
    val_dataset = ImageFolder(root='dataset/tiny-imagenet-200/val', transform=custom_transformer)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = SimpleTinyImageNetNet().to(device)
    model_path = "models/model_epoch3.pt"  # Change this to the desired checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    evaluate(model, val_loader, criterion, device)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()