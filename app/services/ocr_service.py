import os
import importlib.util
from PIL import Image
from app.models.trocr import TrOCRModel
from app.utils.image_utils import preprocess_image

class OCRService:
    """
    OCR service that provides both TrOCR and Tesseract OCR options
    """
    
    def __init__(self):
        """
        Initialize OCR service with both TrOCR and Tesseract OCR
        """
        # Initialize TrOCR model for Khmer language
        self.trocr_model = TrOCRModel()
        
        # Check if pytesseract is installed
        self.pytesseract_installed = importlib.util.find_spec("pytesseract") is not None
        self.tesseract_khmer_available = False
        
        # Only try to import pytesseract if it's installed
        if self.pytesseract_installed:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                
                # Check if Khmer language data is available for Tesseract
                languages = pytesseract.get_languages()
                self.tesseract_khmer_available = 'khm' in languages
            except Exception as e:
                print(f"Error initializing Tesseract: {e}")
    
    def process_image(self, image_path, ocr_engine="trocr"):
        """
        Process an image with OCR
        
        Args:
            image_path (str): Path to the image file
            ocr_engine (str): OCR engine to use ('trocr' or 'tesseract')
            
        Returns:
            str: Detected text in the image
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = preprocess_image(image)
            
            # Perform OCR with selected engine
            if ocr_engine.lower() == "tesseract":
                return self._process_with_tesseract(processed_image)
            else:
                # Default to TrOCR
                return self._process_with_trocr(processed_image)
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
        finally:
            # Clean up the file after processing
            if os.path.exists(image_path):
                os.remove(image_path)
    
    def _process_with_trocr(self, image):
        """Process image with TrOCR"""
        return self.trocr_model.recognize_text(image)
    
    def _process_with_tesseract(self, image):
        """Process image with Tesseract OCR"""
        # Check if pytesseract is installed
        if not self.pytesseract_installed:
            return "Tesseract OCR (pytesseract) is not installed. Please install it first using:\npip install pytesseract"
        
        # Check if Khmer language is available
        if not self.tesseract_khmer_available:
            installation_instructions = """
Khmer language pack not installed for Tesseract. Please install it with:

For Ubuntu/Debian:
sudo apt-get install tesseract-ocr-khm

For macOS:
brew install tesseract-lang

For Windows:
1. Download Khmer traineddata from: https://github.com/tesseract-ocr/tessdata/raw/main/khm.traineddata
2. Place it in the Tesseract tessdata directory (e.g., C:\\Program Files\\Tesseract-OCR\\tessdata)
            """
            return installation_instructions
        
        # Use Tesseract with Khmer language
        # --psm 6 assumes a single uniform block of text
        # -l khm specifies Khmer language
        text = self.pytesseract.image_to_string(
            image, 
            lang='khm',
            config='--psm 6'
        )
        
        return text

    def get_available_engines(self):
        """
        Get available OCR engines
        
        Returns:
            dict: Available OCR engines and their status
        """
        engines = {
            "trocr": True,  # TrOCR is always available since it's bundled
            "tesseract": self.pytesseract_installed and self.tesseract_khmer_available
        }
        
        return engines