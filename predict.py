import torch
import torchvision.models as models
import torch.nn as nn
import argparse
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Argument parser
parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")

parser.add_argument('image_path', type=str, help="Path to the image")
parser.add_argument('checkpoint', type=str, help="Path to checkpoint file")
parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes")
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Path to category names JSON file")
parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

args = parser.parse_args()

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif checkpoint['arch'] == "resnet18":
        model = models.resnet18(pretrained=True)
        input_size = 512

    classifier = nn.Sequential(
        nn.Linear(input_size, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    if checkpoint['arch'] == "vgg16":
        model.classifier = classifier
    elif checkpoint['arch'] == "resnet18":
        model.fc = classifier

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model = load_checkpoint(args.checkpoint)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Process image
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

image_tensor = process_image(args.image_path).to(device)

# Predict class
with torch.no_grad():
    output = model(image_tensor)
probabilities = torch.exp(output)
top_probs, top_indices = probabilities.topk(args.top_k)

top_probs = top_probs.cpu().numpy()[0]
top_indices = top_indices.cpu().numpy()[0]

idx_to_class = {v: k for k, v in model.class_to_idx.items()}
top_classes = [idx_to_class[i] for i in top_indices]

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
top_labels = [cat_to_name[str(cls)] for cls in top_classes]

# Print 
for i in range(args.top_k):
    print(f"{top_labels[i]}: {top_probs[i]:.3f}")
