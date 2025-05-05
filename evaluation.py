import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class Evaluator:
    def __init__(self, class_names=None):
        """
        Initialize the evaluator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names if class_names is not None else ["Class 0", "Class 1", "Class 2"]
    
    def calculate_metrics(self, true_labels, predictions):
        """
        Calculate performance metrics
        
        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print calculated metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        print(f"Overall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['per_class_precision'][i]:.4f}")
            print(f"    Recall:    {metrics['per_class_recall'][i]:.4f}")
            print(f"    F1 Score:  {metrics['per_class_f1'][i]:.4f}")
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """
        Plot confusion matrix
        
        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        return plt.gcf()
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: Dictionary containing training history
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def evaluate_classifier(self, classifier, features, labels, batch_size=32):
        """
        Evaluate a trained classifier
        
        Args:
            classifier: Trained classifier object
            features: Test features
            labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        # Convert numpy arrays to PyTorch tensors
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        # Create dataset and data loader
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        # Get loss and accuracy using the classifier's evaluate method
        loss, accuracy = classifier.evaluate(dataloader)
        
        # Get predictions for detailed metrics
        predictions, _ = classifier.predict(features)
        
        # Calculate detailed metrics
        metrics = self.calculate_metrics(labels, predictions)
        metrics['loss'] = loss
        
        return metrics