import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CustomCNN(nn.Module):
    def __init__(self, input_dim=1280, num_classes=3):
        """
        Custom CNN classifier from scratch
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
        """
        super(CustomCNN, self).__init__()
        
        # Reshape input features to 2D grid for CNN processing
        # Assuming input_dim can be reshaped to a square grid
        self.side_length = int(np.sqrt(input_dim))
        if self.side_length**2 != input_dim:
            # If not a perfect square, pad to the next perfect square
            self.side_length = int(np.ceil(np.sqrt(input_dim)))
            self.padded_dim = self.side_length**2
        else:
            self.padded_dim = input_dim
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions and pooling
        final_size = self.side_length // 8  # After 3 max poolings (2^3 = 8)
        if final_size == 0:  # Handle small input cases
            final_size = 1
        
        # Define fully connected layers
        self.fc1 = nn.Linear(128 * final_size * final_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Pad input if necessary
        if x.shape[1] < self.padded_dim:
            padding = torch.zeros(x.shape[0], self.padded_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape to 2D grid
        x = x.view(-1, 1, self.side_length, self.side_length)
        
        # Apply convolutions with batch normalization, ReLU, and max pooling
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        
        # Calculate the size after convolutions and pooling
        final_size = self.side_length // 8  # After 3 max poolings (2^3 = 8)
        if final_size == 0:  # Handle small input cases
            final_size = 1
        
        # Flatten
        x = x.view(-1, 128 * final_size * final_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class Classifier:
    def __init__(self, input_dim=1280, num_classes=3, learning_rate=0.001):
        """
        Initialize the classifier
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            learning_rate: Learning rate for optimization
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize the model
        self.model = CustomCNN(input_dim, num_classes)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def train(self, features, labels, batch_size=32, epochs=50, validation_data=None):
        """
        Train the classifier
        
        Args:
            features: Training features [num_samples, input_dim]
            labels: Training labels [num_samples]
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_data: Tuple of (val_features, val_labels)
            
        Returns:
            Training history
        """
        # Convert numpy arrays to PyTorch tensors
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        
        # Create dataset and data loader
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Create validation dataloader if validation data is provided
        if validation_data is not None:
            val_features, val_labels = validation_data
            val_features = torch.FloatTensor(val_features)
            val_labels = torch.LongTensor(val_labels)
            val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            for batch_features, batch_labels in dataloader:
                # Move batch to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track loss and predictions
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(batch_labels.cpu().numpy())
            
            # Calculate training metrics
            train_loss /= len(dataloader)
            train_acc = accuracy_score(train_true, train_preds)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        return history
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on a dataloader
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            loss: Average loss
            accuracy: Accuracy score
        """
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                # Move batch to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # Track loss and predictions
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        val_loss /= len(dataloader)
        val_acc = accuracy_score(val_true, val_preds)
        
        return val_loss, val_acc
    
    def predict(self, features):
        """
        Make predictions on new data
        
        Args:
            features: Input features [num_samples, input_dim]
            
        Returns:
            predictions: Predicted class labels
            probabilities: Class probabilities
        """
        # Convert to PyTorch tensor
        features = torch.FloatTensor(features).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy(), probabilities.cpu().numpy()
    
    def save_model(self, path):
        """Save model weights to disk"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights from disk"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()