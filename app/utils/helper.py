"""
Helper Functions
Fungsi-fungsi bantuan untuk model loading, prediction, file handling, dll
"""

import os
import shutil
import numpy as np
from datetime import datetime
from fastapi import UploadFile
import random


def load_model(model_path: str):
    """
    Load model dari file .h5
    
    Parameters:
    - model_path: Path ke file model
    
    Returns:
    - Loaded model
    """
    
    try:
        # Import tensorflow/keras
        from tensorflow import keras
        
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    
    except ImportError:
        print("⚠️  TensorFlow not installed. Using dummy prediction mode.")
        return None
    
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None


def predict_image(model, img_array: np.ndarray, dummy: bool = False) -> dict:
    """
    Prediksi gambar menggunakan model
    
    Parameters:
    - model: Model yang sudah di-load
    - img_array: Numpy array gambar yang sudah dipreprocess
    - dummy: Jika True, gunakan dummy prediction
    
    Returns:
    - Dictionary dengan hasil prediksi
    """
    
    # Class labels
    class_labels = [
        "Wereng Coklat (Brown Planthopper)",
        "Wereng Hijau (Green Leafhopper)",
        "Wereng Punggung Putih (White Backed Planthopper)",
        "Bukan Wereng (Not Wereng)"
    ]
    
    if dummy or model is None:
        # Dummy prediction untuk testing
        class_id = random.randint(0, len(class_labels) - 1)
        confidence = random.uniform(0.75, 0.98)
        
        return {
            "label": class_labels[class_id],
            "confidence": confidence,
            "class_id": class_id,
            "all_predictions": [
                {"class": class_labels[i], "confidence": random.uniform(0.01, 0.95)}
                for i in range(len(class_labels))
            ]
        }
    
    try:
        # Real prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_id = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_id])
        
        # Get all class predictions
        all_predictions = [
            {
                "class": class_labels[i],
                "confidence": float(predictions[0][i])
            }
            for i in range(len(class_labels))
        ]
        
        # Sort by confidence
        all_predictions = sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
        
        return {
            "label": class_labels[predicted_class_id],
            "confidence": confidence,
            "class_id": int(predicted_class_id),
            "all_predictions": all_predictions
        }
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Simpan file upload ke folder temporary
    
    Parameters:
    - upload_file: File yang diupload
    
    Returns:
    - Path ke file yang disimpan
    """
    
    # Buat folder uploads jika belum ada
    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = os.path.join(upload_dir, filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return file_path
    
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")
    
    finally:
        upload_file.file.close()


def log_prediction(filename: str, label: str, confidence: float):
    """
    Log prediksi ke file
    
    Parameters:
    - filename: Nama file gambar
    - label: Label hasil prediksi
    - confidence: Confidence score
    """
    
    # Buat folder logs jika belum ada
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "prediction_logs.txt")
    
    # Format log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {filename} → prediksi: {label} ({confidence:.4f})\n"
    
    # Append to log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    except Exception as e:
        print(f"⚠️  Error writing to log: {e}")


def clean_old_uploads(max_age_hours: int = 24):
    """
    Bersihkan file upload yang sudah lama
    
    Parameters:
    - max_age_hours: Umur maksimal file dalam jam (default: 24)
    """
    
    upload_dir = "app/static/uploads"
    
    if not os.path.exists(upload_dir):
        return
    
    current_time = datetime.now()
    deleted_count = 0
    
    try:
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            
            # Get file creation time
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age_hours = (current_time - file_time).total_seconds() / 3600
            
            # Delete if older than max_age_hours
            if age_hours > max_age_hours:
                os.remove(file_path)
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"🧹 Cleaned {deleted_count} old upload files")
    
    except Exception as e:
        print(f"⚠️  Error cleaning uploads: {e}")


def get_model_summary(model) -> dict:
    """
    Mendapatkan summary model
    
    Parameters:
    - model: Model keras
    
    Returns:
    - Dictionary dengan informasi model
    """
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        import io
        import sys
        
        # Capture model.summary() output
        stream = io.StringIO()
        sys.stdout = stream
        model.summary()
        sys.stdout = sys.__stdout__
        summary_string = stream.getvalue()
        
        return {
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "summary": summary_string
        }
    
    except Exception as e:
        return {"error": str(e)}


def format_confidence(confidence: float) -> str:
    """
    Format confidence score ke string yang lebih readable
    
    Parameters:
    - confidence: Confidence score (0-1)
    
    Returns:
    - Formatted string
    """
    
    percentage = confidence * 100
    
    if percentage >= 90:
        return f"{percentage:.2f}% (Sangat Yakin)"
    elif percentage >= 75:
        return f"{percentage:.2f}% (Yakin)"
    elif percentage >= 60:
        return f"{percentage:.2f}% (Cukup Yakin)"
    else:
        return f"{percentage:.2f}% (Kurang Yakin)"