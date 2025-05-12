from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def preprocess_image(image):
    """
    Preprocess an image for better OCR results
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Processed image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large or too small
    # TrOCR works best with images of reasonable size
    max_size = 1280
    min_size = 320
    
    width, height = image.size
    
    # Resize if needed while maintaining aspect ratio
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    elif min(width, height) < min_size:
        ratio = min_size / min(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Enhance contrast for better text recognition
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    return image

def deskew_image(image):
    """
    Deskew (straighten) an image for better OCR
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Deskewed image
    """
    try:
        # Convert to grayscale
        gray = image.convert('L')
        
        # Convert to numpy array
        gray_array = np.array(gray)
        
        # Find all non-zero points
        coords = np.column_stack(np.where(gray_array > 0))
        
        # Get the minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate the image
        return image.rotate(angle, resample=Image.BICUBIC, expand=True)
    except:
        # If deskewing fails, return the original image
        return image