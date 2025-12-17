import albumentations as A
import cv2
import numpy as np

def get_augmentations():
    """Get data augmentation pipeline for better face matching"""
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=15, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        
        # Color transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        
        # Noise and blur
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Weather effects (simulate different CCTV conditions)
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomShadow(p=0.1),
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.1),
        A.RandomRain(p=0.1),
        
        # Image quality
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.2),
    ])

def apply_augmentations(image, augmentations):
    """Apply augmentations to image"""
    if augmentations is None:
        return image
    
    augmented = augmentations(image=image)
    return augmented['image']