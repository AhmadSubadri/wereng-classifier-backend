"""
Classification Route
Endpoint untuk klasifikasi gambar hama wereng
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
from typing import Optional

from app.utils.preprocessing import preprocess_image
from app.utils.helper import load_model, predict_image, save_upload_file, log_prediction

router = APIRouter()

# Load model saat startup
MODEL_PATH = "app/models/wereng_classifier.h5"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  Model file not found. Using dummy prediction mode.")
except Exception as e:
    print(f"⚠️  Error loading model: {e}. Using dummy prediction mode.")


@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Endpoint untuk klasifikasi gambar hama wereng
    
    Parameters:
    - file: Image file (JPG, JPEG, PNG)
    
    Returns:
    - JSON dengan hasil prediksi
    """
    
    # Validasi tipe file
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Simpan file upload sementara
        file_path = await save_upload_file(file)
        
        # Preprocess gambar
        processed_image = preprocess_image(file_path)
        
        # Prediksi
        if model is not None:
            # Gunakan model real
            prediction_result = predict_image(model, processed_image)
        else:
            # Gunakan dummy prediction
            prediction_result = predict_image(None, processed_image, dummy=True)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "label": prediction_result["label"],
                "confidence": round(prediction_result["confidence"], 4),
                "class_id": prediction_result.get("class_id", 0)
            },
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log prediction
        log_prediction(
            filename=file.filename,
            label=prediction_result["label"],
            confidence=prediction_result["confidence"]
        )
        
        # Hapus file temporary
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/classify/batch")
async def classify_batch_images(files: list[UploadFile] = File(...)):
    """
    Endpoint untuk klasifikasi batch (multiple images)
    
    Parameters:
    - files: List of image files
    
    Returns:
    - JSON dengan hasil prediksi untuk setiap gambar
    """
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )
    
    results = []
    
    for file in files:
        # Validasi tipe file
        allowed_extensions = [".jpg", ".jpeg", ".png"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "File type not supported"
            })
            continue
        
        try:
            # Simpan file upload sementara
            file_path = await save_upload_file(file)
            
            # Preprocess gambar
            processed_image = preprocess_image(file_path)
            
            # Prediksi
            if model is not None:
                prediction_result = predict_image(model, processed_image)
            else:
                prediction_result = predict_image(None, processed_image, dummy=True)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "label": prediction_result["label"],
                    "confidence": round(prediction_result["confidence"], 4),
                    "class_id": prediction_result.get("class_id", 0)
                }
            })
            
            # Log prediction
            log_prediction(
                filename=file.filename,
                label=prediction_result["label"],
                confidence=prediction_result["confidence"]
            )
            
            # Hapus file temporary
            if os.path.exists(file_path):
                os.remove(file_path)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }