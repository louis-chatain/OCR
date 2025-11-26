from fastapi import FastAPI, File, HTTPException, UploadFile
import asyncio
import os
import tempfile
import pytesseract

# -----------------------------------------------------------------------------------------
# Source - https://stackoverflow.com/a
# Posted by R. Marolahy
# Retrieved 2025-11-26, License - CC BY-SA 4.0

pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
# -----------------------------------------------------------------------------------------

app = FastAPI(title="Safe OCR Service")

# --- Helper function to handle the blocking OCR process ---
# We wrap the synchronous, blocking pytesseract call in a standard 'def' 
# function. FastAPI will automatically run this in a separate thread 
# (using the internal ThreadPoolExecutor) when called from the async endpoint.
def perform_ocr_sync(filepath: str) -> str:
    """Synchronously performs OCR on a local file path."""
    try:
        # Use lang='eng' as specified in the original request
        return pytesseract.image_to_string(filepath, lang="eng")
    except pytesseract.TesseractNotFoundError:
        # Tesseract must be installed and in the PATH for this to work
        print("Tesseract not found. Please ensure it is installed and in your PATH.")
        # Re-raise as an application error
        raise RuntimeError("Tesseract OCR engine is not configured correctly.")
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        raise RuntimeError(f"OCR processing failed: {e}")


@app.post("/ocr", response_model=dict, summary="Perform OCR on an uploaded image")
async def ocr_endpoint(image: UploadFile = File(..., description="Image file (PNG, JPG, etc.) for OCR")):
    """
    Receives an image, saves it to a secure temporary file, 
    performs OCR, and returns the extracted text.
    """
    if not image.content_type or not image.content_type.startswith('image/'):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
         
    # 1. Use NamedTemporaryFile for safe, concurrent file handling.
    # 'delete=True' ensures the file is removed when the context manager exits.
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
            # 2. Read the uploaded file contents asynchronously in chunks
            # This prevents blocking the main event loop while receiving the data.
            while content := await image.read(1024 * 1024): # Read in 1MB chunks
                tmp.write(content)
            
            # Flush data to disk to ensure it's fully written before OCR starts
            tmp.flush()
            
            # 3. Get the path and perform the synchronous OCR operation
            filepath = tmp.name
            raw_text = await asyncio.to_thread(perform_ocr_sync, filepath)

    except HTTPException:
        # Re-raise existing HTTP exceptions
        raise 
    except RuntimeError as e:
        # Handle errors from the OCR worker
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Handle any other general file handling error
        print(f"File handling error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file processing.")
    finally:
        # 4. Ensure the temporary file is deleted, even if exceptions occurred.
        # This is a safety check; NamedTemporaryFile with delete=True handles this 
        # on close, but we ensure it's removed by path here too in case of delete=False
        # or if we need to clean up an incomplete file.
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            
    return {"raw_text_from_img": raw_text}
