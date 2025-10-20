"""
Image Preprocessing Utilities
Modul untuk preprocessing gambar sebelum prediksi
"""

import numpy as np
from PIL import Image
import io


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess gambar untuk model CNN
    
    Parameters:
    - image_path: Path ke file gambar
    - target_size: Ukuran target gambar (default: 224x224 untuk MobileNetV2)
    
    Returns:
    - Numpy array gambar yang sudah dipreprocess
    """
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB jika grayscale atau RGBA
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def preprocess_image_from_bytes(image_bytes: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess gambar dari bytes
    
    Parameters:
    - image_bytes: Bytes dari gambar
    - target_size: Ukuran target gambar
    
    Returns:
    - Numpy array gambar yang sudah dipreprocess
    """
    
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image from bytes: {str(e)}")


def augment_image(img_array: np.ndarray, augmentation_type: str = "none") -> np.ndarray:
    """
    Augmentasi gambar untuk meningkatkan robustness
    
    Parameters:
    - img_array: Numpy array gambar
    - augmentation_type: Tipe augmentasi (flip, rotate, brightness, none)
    
    Returns:
    - Augmented image array
    """
    
    # Remove batch dimension temporarily
    img = img_array[0]
    
    if augmentation_type == "flip":
        img = np.fliplr(img)
    elif augmentation_type == "rotate":
        img = np.rot90(img)
    elif augmentation_type == "brightness":
        # Increase brightness slightly
        img = np.clip(img * 1.2, 0, 1)
    
    # Add batch dimension back
    return np.expand_dims(img, axis=0)


def validate_image(image_path: str) -> dict:
    """
    Validasi gambar (format, ukuran, dll)
    
    Parameters:
    - image_path: Path ke file gambar
    
    Returns:
    - Dictionary dengan informasi validasi
    """
    
    try:
        img = Image.open(image_path)
        
        return {
            "valid": True,
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height
        }
    
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def get_image_info(image_path: str) -> dict:
    """
    Mendapatkan informasi detail gambar
    
    Parameters:
    - image_path: Path ke file gambar
    
    Returns:
    - Dictionary dengan informasi gambar
    """
    
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1,
            "dtype": str(img_array.dtype),
            "min_pixel": int(img_array.min()),
            "max_pixel": int(img_array.max()),
            "mean_pixel": float(img_array.mean())
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }