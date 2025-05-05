import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import random

class Preprocessor:
    def __init__(self, target_size=(224, 224), augment=True):
        self.target_size = target_size
        self.augment = augment
        
        # Define augmentation pipeline for training
        self.train_transforms = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            ], p=0.5),
            A.OneOf([
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Define validation transforms (no augmentation)
        self.val_transforms = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def preprocess(self, image, is_training=True):
        """
        Preprocess an input image with optional augmentation
        
        Args:
            image: Input image (numpy array)
            is_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image as a tensor
        """
        if is_training and self.augment:
            transformed = self.train_transforms(image=image)
        else:
            transformed = self.val_transforms(image=image)
            
        return transformed["image"]
    
    def apply_thresholding(self, image):
        """Apply adaptive thresholding to an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh
    
    def apply_morphological_operations(self, binary_image):
        """Apply morphological operations to a binary image"""
        kernel = np.ones((5, 5), np.uint8)
        # Opening (erosion followed by dilation)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        # Closing (dilation followed by erosion)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing
    
    def noise_reduction(self, image):
        """Apply noise reduction to an image"""
        # Bilateral filtering preserves edges while reducing noise
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def enhance_contrast(self, image):
        """Enhance contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def apply_sharpening(self, image):
        """Apply sharpening filter to an image"""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def full_preprocessing(self, image, is_training=True):
        """Apply complete preprocessing pipeline with optional enhancements"""
        # Convert to numpy array if it's PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Apply basic enhancements
        denoised = self.noise_reduction(image)
        enhanced = self.enhance_contrast(denoised)
        
        # Sometimes apply sharpening (randomly during training)
        if is_training and random.random() < 0.3:
            enhanced = self.apply_sharpening(enhanced)
            
        # Apply augmentation and normalization
        processed = self.preprocess(enhanced, is_training)
        
        return processed