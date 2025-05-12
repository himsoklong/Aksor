from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from pathlib import Path
import shutil
from app.services.ocr_service import OCRService
from typing import Optional

router = APIRouter()
ocr_service = OCRService()

@router.post("/ocr")
async def process_image(
    file: UploadFile = File(...),
    ocr_engine: Optional[str] = Form("trocr")  # Default to TrOCR
):
    """
    Process an image with OCR
    
    Args:
        file: Image file to process
        ocr_engine: OCR engine to use ('trocr' or 'tesseract')
    
    Returns:
        JSON response with detected text
    """
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate OCR engine
    if ocr_engine.lower() not in ["trocr", "tesseract"]:
        raise HTTPException(status_code=400, detail="Invalid OCR engine. Use 'trocr' or 'tesseract'")
    
    # Check if tesseract is available when selected
    available_engines = ocr_service.get_available_engines()
    if ocr_engine.lower() == "tesseract" and not available_engines["tesseract"]:
        raise HTTPException(
            status_code=400, 
            detail="Tesseract OCR with Khmer language support is not available on this server"
        )
    
    # Create unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"uploads/{unique_filename}"
    
    # Create uploads directory if it doesn't exist
    Path("uploads").mkdir(exist_ok=True)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()
    
    # Process image with OCR
    try:
        text = ocr_service.process_image(file_path, ocr_engine)
        return {
            "filename": file.filename,
            "detected_text": text,
            "engine_used": ocr_engine
        }
    except Exception as e:
        # Clean up file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@router.get("/engines")
async def get_available_engines():
    """
    Get available OCR engines
    
    Returns:
        JSON response with available OCR engines
    """
    return ocr_service.get_available_engines()