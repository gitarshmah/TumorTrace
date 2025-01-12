import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Check if the train directory exists
train_dir = "C:/Users/Darshan GK/OneDrive/Desktop/exta/TumorTrace/data/train"

if not os.path.exists(train_dir):
    print(f"Error: '{train_dir}' directory does not exist.")
else:
    # Step 2: Check if there are class subfolders in the train directory
    subfolders = [f.path for f in os.scandir(train_dir) if f.is_dir()]
    
    if len(subfolders) == 0:
        print(f"Error: No class subfolders found in {train_dir}.")
    else:
        # Step 3: Define the transformations to be applied to the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),  # Convert the images to tensor format
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        ])

        # Step 4: Load the train dataset using ImageFolder
        train_data = datasets.ImageFolder(root=train_dir, transform=transform)

        # Step 5: Print out the number of images found in the train dataset
        print(f"Found {len(train_data)} images in {train_dir}.")

        # Step 6: Define a DataLoader for batching and shuffling the dataset
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Optional: Sample a batch of data to check if everything works
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"Sample batch: {images.shape}, {labels.shape}")

# Now you can add your model training code here, e.g., defining a model, loss function, optimizer, and running the training loop.
