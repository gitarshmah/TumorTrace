import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Import your model architecture here
from model import TumorClassifier

# Data Preprocessing
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])

test_dataset = datasets.ImageFolder(root="data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the saved model
checkpoint = torch.load("models/model_checkpoint.pth")
model = TumorClassifier()  # Replace with your model class
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()  # Use GPU if available
model.eval()

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy of the model on the test dataset: {accuracy * 100:.2f}%")
