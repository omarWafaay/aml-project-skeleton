import os
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter
import wandb

from utils.load_transformed import custom_transformer
from utils.train_models import SimpleTinyImageNetNet


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc, "Epoch": epoch})


def validate(model, val_loader, criterion, device):
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
    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})
    return accuracy


def main():
    # Initialize wandb
    # Initialize wandb
    wandb.login(relogin=True)

    wandb.init(project="tiny-imagenet", name="simple-cnn-run")

    # Load transformed data
    tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet-200/train', transform=custom_transformer)
    tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet-200/val', transform=custom_transformer)

    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    train_loader = DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=1)
    val_loader = DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = SimpleTinyImageNetNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0.0
    num_epochs = 3

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(model, val_loader, criterion, device)

        # Save model checkpoint
        model_path = f"models/model_epoch{epoch}.pt"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        best_acc = max(best_acc, val_accuracy)
        print(f"\n✅ Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()