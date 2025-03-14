import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
import json
import os

parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset of images")
parser.add_argument('data_dir', type=str, help="Path to dataset")
parser.add_argument('--save_dir', type=str, default='.', help="Directory to save checkpoints")
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help="Model architecture")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units")
parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training if available")

args = parser.parse_args()
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

train_dir = os.path.join(args.data_dir, 'train')
valid_dir = os.path.join(args.data_dir, 'valid')


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False)
}

# Load pre-trained model
if args.arch == "vgg16":
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == "resnet18":
    model = models.resnet18(pretrained=True)
    input_size = 512

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

if args.arch == "vgg16":
    model.classifier = classifier
elif args.arch == "resnet18":
    model.fc = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

print("Starting training...")

for epoch in range(args.epochs):
    running_loss = 0
    model.train()

    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation step
    model.eval()
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()

            # Accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train Loss: {running_loss/len(dataloaders['train']):.3f}.. "
          f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
          f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# Save checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'arch': args.arch,
    'hidden_units': args.hidden_units
}
torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
print("Training complete. Model saved!")
