"""
Model Info Route
Endpoint untuk informasi model dan history
"""

from fastapi import APIRouter
from datetime import datetime
import os
import json

router = APIRouter()

# Model metadata (hardcoded untuk demo, bisa diganti dari file JSON)
MODEL_METADATA = {
    "model_name": "Wereng Classifier",
    "model_version": "1.0.0",
    "architecture": "MobileNetV2",
    "input_shape": [224, 224, 3],
    "total_classes": 4,
    "classes": [
        "Wereng Coklat (Brown Planthopper)",
        "Wereng Hijau (Green Leafhopper)",
        "Wereng Punggung Putih (White Backed Planthopper)",
        "Bukan Wereng (Not Wereng)"
    ],
    "training_accuracy": 0.95,
    "validation_accuracy": 0.92,
    "training_date": "2025-10-15",
    "dataset_size": 2000,
    "epochs": 50
}


@router.get("/model/info")
async def get_model_info():
    """
    Mendapatkan informasi tentang model yang digunakan
    
    Returns:
    - Metadata model (akurasi, label, tanggal training, dll)
    """
    
    model_exists = os.path.exists("app/models/wereng_classifier.h5")
    
    return {
        "success": True,
        "model_loaded": model_exists,
        "model_info": MODEL_METADATA,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/model/classes")
async def get_model_classes():
    """
    Mendapatkan daftar kelas yang dapat diprediksi oleh model
    
    Returns:
    - List of classes
    """
    
    return {
        "success": True,
        "total_classes": MODEL_METADATA["total_classes"],
        "classes": [
            {
                "id": idx,
                "name": class_name,
                "description": get_class_description(class_name)
            }
            for idx, class_name in enumerate(MODEL_METADATA["classes"])
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/history")
async def get_prediction_history(limit: int = 50):
    """
    Mendapatkan riwayat prediksi
    
    Parameters:
    - limit: Maximum number of records to return (default: 50)
    
    Returns:
    - List of prediction history
    """
    
    log_file = "logs/prediction_logs.txt"
    
    if not os.path.exists(log_file):
        return {
            "success": True,
            "total_records": 0,
            "history": [],
            "message": "No prediction history available",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse log lines
        history = []
        for line in reversed(lines[-limit:]):  # Get last N records
            if "→" in line:
                parts = line.strip().split("→")
                if len(parts) == 2:
                    timestamp_filename = parts[0].strip()
                    prediction_info = parts[1].strip()
                    
                    # Parse timestamp dan filename
                    timestamp = timestamp_filename.split("]")[0].replace("[", "").strip()
                    filename = timestamp_filename.split("]")[1].strip()
                    
                    # Parse prediction
                    label = prediction_info.split("(")[0].replace("prediksi:", "").strip()
                    confidence = prediction_info.split("(")[1].replace(")", "").strip()
                    
                    history.append({
                        "timestamp": timestamp,
                        "filename": filename,
                        "prediction": label,
                        "confidence": float(confidence)
                    })
        
        return {
            "success": True,
            "total_records": len(history),
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.delete("/history")
async def clear_prediction_history():
    """
    Menghapus riwayat prediksi
    
    Returns:
    - Confirmation message
    """
    
    log_file = "logs/prediction_logs.txt"
    
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
            message = "Prediction history cleared successfully"
        else:
            message = "No prediction history to clear"
        
        return {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_class_description(class_name: str) -> str:
    """
    Helper function untuk mendapatkan deskripsi kelas
    """
    descriptions = {
        "Wereng Coklat (Brown Planthopper)": "Hama paling merusak pada tanaman padi, menyebabkan hopperburn",
        "Wereng Hijau (Green Leafhopper)": "Vektor virus tungro, menyebabkan daun menguning",
        "Wereng Punggung Putih (White Backed Planthopper)": "Menghisap cairan tanaman, menyebabkan daun mengering",
        "Bukan Wereng (Not Wereng)": "Bukan termasuk hama wereng atau objek lain"
    }
    return descriptions.get(class_name, "No description available")