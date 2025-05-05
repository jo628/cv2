import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

def load_dataset_info(data_dir, classes=None, split_file=None):
    """
    Load dataset information from MINC-2500
    
    Args:
        data_dir: Path to the MINC-2500 dataset directory
        classes: List of classes to include (default: all)
        split_file: Path to a specific split file (default: None)
        
    Returns:
        List of (image_path, label) tuples
    """
    if split_file is None:
        # Default to train1.txt
        split_file = os.path.join(data_dir, 'labels', 'train1.txt')
    
    # Load category indices and names
    categories = []
    with open(os.path.join(data_dir, 'categories.txt'), 'r') as f:
        for line in f:
            category = line.strip()
            categories.append(category)
    
    # Filter categories if classes is specified
    if classes is not None:
        valid_categories = [c for c in categories if c in classes]
        category_to_idx = {category: idx for idx, category in enumerate(valid_categories)}
    else:
        category_to_idx = {category: idx for idx, category in enumerate(categories)}
    
    # Load image paths and labels from split file
    dataset_info = []
    with open(split_file, 'r') as f:
        for line in f:
            # Format is: images/category/image_file.jpg
            relative_path = line.strip()
            
            # Extract category name from path
            category = relative_path.split('/')[1]
            
            # Skip if category not in our target classes
            if classes is not None and category not in classes:
                continue
            
            # Get absolute path
            image_path = os.path.join(data_dir, relative_path)
            
            # Get label index
            label = category_to_idx[category]
            
            dataset_info.append((image_path, label))
    
    return dataset_info, category_to_idx

class RecycledMaterialsDataset(Dataset):
    """Dataset class for recycled materials"""
    
    def __init__(self, dataset_info, transform=None):
        """
        Initialize the dataset
        
        Args:
            dataset_info: List of (image_path, label) tuples
            transform: Optional transform to apply to the images
        """
        self.dataset_info = dataset_info
        self.transform = transform
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.dataset_info)
    
    def __getitem__(self, idx):
        """Get an item from the dataset"""
        image_path, label = self.dataset_info[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataloaders(dataset_info, transform, batch_size=32, split_ratio=0.8, seed=42):
    """
    Create training and validation dataloaders
    
    Args:
        dataset_info: List of (image_path, label) tuples
        transform: Transform object for preprocessing
        batch_size: Batch size for the dataloaders
        split_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the dataset
    dataset_info = dataset_info.copy()
    random.shuffle(dataset_info)
    
    # Split into training and validation sets
    split_idx = int(len(dataset_info) * split_ratio)
    train_info = dataset_info[:split_idx]
    val_info = dataset_info[split_idx:]
    
    # Create datasets
    train_dataset = RecycledMaterialsDataset(train_info, transform)
    val_dataset = RecycledMaterialsDataset(val_info, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def visualize_images(dataloader, num_images=5, classes=None):
    """
    Visualize random images from a dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        num_images: Number of images to visualize
        classes: List of class names
    """
    images, labels = next(iter(dataloader))
    
    # Convert from tensor to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.numpy()
        
        # If images are normalized, denormalize them
        if images.max() <= 1.0:
            images = images * 255
        
        # Move channel dimension to the end for plotting
        images = np.transpose(images, (0, 2, 3, 1))
    
    # Limit to num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Plot images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img.astype(np.uint8))
        if classes:
            axes[i].set_title(f"Class: {classes[label]}")
        else:
            axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_segmentation(original, segmented, masked=None, title=None):
    """
    Visualize original, segmented, and masked images side by side
    
    Args:
        original: Original image
        segmented: Segmentation mask
        masked: Masked image (optional)
        title: Plot title (optional)
    """
    n_images = 3 if masked is not None else 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, n_images, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, n_images, 2)
    plt.imshow(segmented, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    if masked is not None:
        plt.subplot(1, n_images, 3)
        plt.imshow(masked)
        plt.title('Masked Image')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
        
    plt.tight_layout()
    plt.show()