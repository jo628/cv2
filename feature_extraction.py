import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class FeatureExtractor:
    def __init__(self, model_name='mobilenet_v2', use_pretrained=True, output_dim=1280):
        """
        Initialize the feature extractor
        
        Args:
            model_name: Name of the model to use
            use_pretrained: Whether to use pretrained weights
            output_dim: Output dimension of the feature vectors
        """
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Initialize feature extraction model
        if model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=use_pretrained)
            # Remove the classifier to get features only
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=use_pretrained)
            # Remove the classifier to get features only
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def extract_features(self, images):
        """
        Extract features from preprocessed images
        
        Args:
            images: Tensor of preprocessed images [batch_size, 3, H, W]
            
        Returns:
            Feature vectors [batch_size, output_dim]
        """
        with torch.no_grad():
            # Forward pass through the model
            features = self.model(images)
            # Reshape features to [batch_size, channels]
            features = features.reshape(features.size(0), -1)
            
        return features
    
    def _load_model_to_device(self, device):
        """Move model to the specified device"""
        self.model = self.model.to(device)
    
    def batch_extract_features(self, data_loader, device='cuda'):
        """
        Extract features for all images in a data loader
        
        Args:
            data_loader: PyTorch DataLoader containing images
            device: Device to use for computation
            
        Returns:
            features: Numpy array of features for all images
            labels: Numpy array of corresponding labels
        """
        # Move model to device
        self._load_model_to_device(device)
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_images, batch_labels in data_loader:
                # Move batch to device
                batch_images = batch_images.to(device)
                
                # Extract features
                batch_features = self.extract_features(batch_images)
                
                # Move features and labels back to CPU for storage
                features_list.append(batch_features.cpu().numpy())
                labels_list.append(batch_labels.numpy())
        
        # Concatenate all batches
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        return features, labels