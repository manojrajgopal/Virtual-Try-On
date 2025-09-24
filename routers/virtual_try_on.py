import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image
import io
from typing import Optional
import logging
import asyncio
from routers.token_manager import token_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def validate_image(content: bytes) -> bool:
    """Validate if the content is a valid image"""
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        return True
    except Exception:
        return False

@router.post("/virtual-try-on")
async def virtual_try_on_endpoint(
    vton_image: UploadFile = File(..., description="Person image"),
    garment_image: UploadFile = File(..., description="Garment image")
):
    """
    Virtual Try-On API Endpoint with Proper Request Queuing
    """
    logger.info("Received virtual try-on request")
    
    # Validate file types
    if not allowed_file(vton_image.filename) or not allowed_file(garment_image.filename):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only PNG, JPG, JPEG, and WEBP are allowed."
        )
    
    vton_temp_path = None
    garment_temp_path = None
    
    try:
        # Read and validate images
        vton_content = await vton_image.read()
        garment_content = await garment_image.read()
        
        # Validate that files are actual images
        if not await validate_image(vton_content) or not await validate_image(garment_content):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check service status
        service_status = token_manager.get_service_status()
        if service_status["usable_tokens"] == 0:
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. All tokens have exceeded daily GPU quota. Please try again in 24 hours."
            )
        
        # Check queue size
        if service_status["queue_size"] > 10:  # Increased queue limit
            raise HTTPException(
                status_code=503,
                detail="Service is currently busy. Please try again in a few minutes."
            )
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as vton_temp:
            vton_temp.write(vton_content)
            vton_temp_path = vton_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as garment_temp:
            garment_temp.write(garment_content)
            garment_temp_path = garment_temp.name
        
        logger.info(f"Queueing request. Queue size: {service_status['queue_size']}, Usable tokens: {service_status['usable_tokens']}")
        
        # Process images with token rotation (this will queue the request)
        result = await token_manager.process_with_token(vton_temp_path, garment_temp_path)

        if not result:
            # Check current status
            current_status = token_manager.get_service_status()
            
            if current_status["usable_tokens"] == 0:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable due to quota limitations. Please try again tomorrow."
                )
            else:
                # More specific error message
                raise HTTPException(
                    status_code=500,
                    detail="Processing failed. This could be due to temporary service issues. Please try again with different images."
                )

        # Handle the result
        image_data = await extract_image_data(result)
        
        if image_data:
            logger.info("Virtual try-on completed successfully")
            return Response(
                content=image_data,
                media_type="image/jpeg",
                headers={"Content-Disposition": "attachment; filename=virtual_try_on_result.jpg"}
            )
        else:
            # If we can't extract image data but have a result, try to return it as JSON
            logger.warning("Could not extract image data, returning result info")
            return JSONResponse(
                content={
                    "status": "processed",
                    "message": "Processing completed but could not extract image",
                    "result_type": str(type(result)),
                    "result_preview": str(result)[:200] if result else "None"
                },
                status_code=200
            )

    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(
            status_code=504,
            detail="Request timeout. The service is currently busy. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )
    finally:
        # Clean up temporary files
        if vton_temp_path and os.path.exists(vton_temp_path):
            os.unlink(vton_temp_path)
        if garment_temp_path and os.path.exists(garment_temp_path):
            os.unlink(garment_temp_path)

async def extract_image_data(result) -> Optional[bytes]:
    """Extract image data from various possible result formats"""
    if not result:
        return None
        
    try:
        # If result is a file path
        if isinstance(result, str) and os.path.exists(result):
            with open(result, "rb") as f:
                return f.read()
        
        # If result is a dictionary with image path
        if isinstance(result, dict) and 'image' in result:
            image_path = result['image']
            if isinstance(image_path, str) and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    return f.read()
        
        # If result contains a file path as string
        if isinstance(result, str):
            # Try to find a file path in the string
            import re
            path_match = re.search(r'/(tmp|temp|home|root)/[^\s\']+\.(jpg|jpeg|png|webp)', result)
            if path_match:
                possible_path = path_match.group(0)
                if os.path.exists(possible_path):
                    with open(possible_path, "rb") as f:
                        return f.read()
        
        logger.warning(f"Could not extract image data from result type: {type(result)}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting image data: {str(e)}")
        return None

@router.get("/virtual-try-on/status")
async def get_token_status():
    """Get current token status and queue information"""
    status = token_manager.get_service_status()
    
    response = {
        "service_status": status["service_status"],
        "queue_size": status["queue_size"],
        "tokens": {
            "total": status["total_tokens"],
            "usable": status["usable_tokens"],
            "unusable": status["unusable_tokens"]
        },
        "details": status["token_details"]
    }
    
    # Add advice based on status
    if status["usable_tokens"] == 0:
        response["message"] = "All tokens have exceeded daily quota. Service will resume in 24 hours."
        response["suggestion"] = "Add more Hugging Face tokens to increase capacity."
    elif status["queue_size"] > 0:
        response["message"] = f"Service operational. {status['queue_size']} requests in queue."
        response["estimated_wait"] = f"Approx {status['queue_size'] * 20} seconds"
    else:
        response["message"] = "Service ready and waiting for requests."
    
    return response

@router.get("/virtual-try-on/queue-info")
async def get_queue_info():
    """Get detailed queue information"""
    status = token_manager.get_service_status()
    
    return {
        "queue_size": status["queue_size"],
        "active_operations": [
            op for op in [token_manager.current_operations.get(token) 
                         for token in token_manager.tokens] if op
        ],
        "estimated_wait_time_seconds": status["queue_size"] * 20,
        "service_capacity": f"{status['usable_tokens']} concurrent operations"
    }