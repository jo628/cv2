#!/usr/bin/env python3
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random
from sklearn.model_selection import train_test_split

# Import custom modules
from preprocessing import Preprocessor
from segmentation import Segmenter
from feature_extraction import FeatureExtractor
from classification import Classifier
from evaluation import Evaluator
from utils import load_dataset_info, RecycledMaterialsDataset, create_dataloaders, visualize_images, visualize_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Recycled Materials Classification Pipeline')
    parser.add_argument('--data_dir', type=str, default='/content/minc2500/minc-2500', 
                        help='Path to the MINC-2500 dataset')
    parser.add_argument('--classes', type=str, default='metal,paper,plastic', 
                        help='Comma-separated list of classes to use')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for optimization')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--segmentation_method', type=str, default='threshold', 
                        choices=['threshold', 'edge', 'region', 'kmeans'],
                        help='Segmentation method to use')
    parser.add_argument('--feature_model', type=str, default='mobilenet_v2', 
                        choices=['mobilenet_v2', 'efficientnet_b0'],
                        help='Feature extraction model to use')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize samples from the pipeline')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse command-line arguments
    try:
        args = parse_args()
    except:
        # If run in notebook environment without arguments, use defaults
        class Args:
            data_dir = '/content/minc2500/minc-2500'
            classes = 'metal,paper,plastic'
            batch_size = 32
            epochs = 50
            learning_rate = 0.001
            seed = 42
            segmentation_method = 'threshold'
            feature_model = 'mobilenet_v2'
            output_dir = 'output'
            visualize = True
        args = Args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse classes
    classes = args.classes.split(',')
    print(f"Using classes: {classes}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset information
    print("Loading dataset information...")
    
    # Try different split files until we find one that exists
    split_files = [
        os.path.join(args.data_dir, 'labels', 'train1.txt'),
        os.path.join(args.data_dir, 'labels', 'train2.txt'),
        os.path.join(args.data_dir, 'labels', 'train3.txt'),
        os.path.join(args.data_dir, 'labels', 'train4.txt'),
        os.path.join(args.data_dir, 'labels', 'train5.txt')
    ]
    
    train_split_file = None
    for split_file in split_files:
        if os.path.exists(split_file):
            train_split_file = split_file
            break
    
    if train_split_file is None:
        raise FileNotFoundError("Could not find any train split files")
    
    print(f"Using training split file: {train_split_file}")
    
    # Load training data
    train_info, category_to_idx = load_dataset_info(
        args.data_dir, classes=classes, split_file=train_split_file
    )
    
    # Find a test split file
    test_split_files = [
        os.path.join(args.data_dir, 'labels', 'test1.txt'),
        os.path.join(args.data_dir, 'labels', 'test2.txt'),
        os.path.join(args.data_dir, 'labels', 'test3.txt'),
        os.path.join(args.data_dir, 'labels', 'test4.txt'),
        os.path.join(args.data_dir, 'labels', 'test5.txt')
    ]
    
    test_split_file = None
    for split_file in test_split_files:
        if os.path.exists(split_file):
            test_split_file = split_file
            break
    
    if test_split_file is None:
        # If no test split file is found, use a portion of train data
        print("No test split file found. Using a portion of training data for testing.")
        train_info, test_info = train_test_split(train_info, test_size=0.2, random_state=args.seed)
    else:
        print(f"Using test split file: {test_split_file}")
        test_info, _ = load_dataset_info(
            args.data_dir, classes=classes, split_file=test_split_file
        )
    
    print(f"Train set size: {len(train_info)}")
    print(f"Test set size: {len(test_info)}")
    
    # Initialize pipeline components
    print("Initializing pipeline components...")
    
    # 1. Preprocessing
    preprocessor = Preprocessor(target_size=(224, 224), augment=True)
    
    # 2. Segmentation
    segmenter = Segmenter(method=args.segmentation_method)
    
    # 3. Feature Extraction
    feature_extractor = FeatureExtractor(model_name=args.feature_model, use_pretrained=True)
    
    # 4. Classification
    input_dim = 1280  # Default for MobileNetV2
    num_classes = len(classes)
    classifier = Classifier(input_dim=input_dim, num_classes=num_classes, learning_rate=args.learning_rate)
    
    # 5. Evaluation
    evaluator = Evaluator(class_names=classes)
    
    # Create datasets
    class TrainDataset(RecycledMaterialsDataset):
        def __init__(self, dataset_info, preprocessor, segmenter):
            super().__init__(dataset_info, None)
            self.preprocessor = preprocessor
            self.segmenter = segmenter
        
        def __getitem__(self, idx):
            image_path, label = self.dataset_info[idx]
            
            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing
            processed = self.preprocessor.full_preprocessing(image, is_training=True)
            
            return processed, label
    
    class TestDataset(RecycledMaterialsDataset):
        def __init__(self, dataset_info, preprocessor, segmenter):
            super().__init__(dataset_info, None)
            self.preprocessor = preprocessor
            self.segmenter = segmenter
        
        def __getitem__(self, idx):
            image_path, label = self.dataset_info[idx]
            
            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing without augmentation
            processed = self.preprocessor.full_preprocessing(image, is_training=False)
            
            return processed, label
    
    # Create datasets
    train_dataset = TrainDataset(train_info, preprocessor, segmenter)
    test_dataset = TestDataset(test_info, preprocessor, segmenter)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Visualize some training examples if requested
    if args.visualize:
        print("Visualizing pipeline steps...")
        # Get a sample image
        sample_path, sample_label = train_info[0]
        sample_image = cv2.imread(sample_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        
        # Apply segmentation
        segmentation_mask = segmenter.segment(sample_image)
        masked_image = segmenter.apply_mask(sample_image, segmentation_mask)
        
        # Visualize
        visualize_segmentation(
            sample_image, segmentation_mask, masked_image,
            title=f"Segmentation Example (Class: {classes[sample_label]})"
        )
    
    # Extract features
    print("Extracting features from training set...")
    train_features, train_labels = feature_extractor.batch_extract_features(train_loader, device=device)
    
    print("Extracting features from test set...")
    test_features, test_labels = feature_extractor.batch_extract_features(test_loader, device=device)
    
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # Train classifier
    print("Training classifier...")
    history = classifier.train(
        train_features, train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(test_features, test_labels)
    )
    
    # Save trained model
    model_path = os.path.join(args.output_dir, 'classifier_model.pth')
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluator.evaluate_classifier(classifier, test_features, test_labels, batch_size=args.batch_size)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Plot confusion matrix
    predictions, _ = classifier.predict(test_features)
    cm_fig = evaluator.plot_confusion_matrix(test_labels, predictions)
    cm_fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot training history
    history_fig = evaluator.plot_training_history(history)
    history_fig.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    print(f"Results saved to {args.output_dir}")
    
    # Check if accuracy meets the goal
    if metrics['accuracy'] >= 0.9:
        print("✅ Goal achieved! Accuracy is above 90%")
    else:
        print(f"⚠️ Goal not met. Accuracy is {metrics['accuracy']:.4f}, which is below 90%")

if __name__ == "__main__":
    main()