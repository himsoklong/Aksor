import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

class TrOCRModel:
    def __init__(self, model_path=None):
        """
        Initialize TrOCR model for Khmer OCR
        
        Args:
            model_path (str, optional): Path to fine-tuned model directory. 
                If None, uses the best available pretrained model.
        """
        # Default to pretrained model that works with Khmer
        if model_path is None or not os.path.exists(model_path):
            # You can replace this with a specific fine-tuned model for Khmer
            # This is a starting point - the microsoft/trocr-base-handwritten model
            # We'll fine-tune this for Khmer
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        else:
            # Load fine-tuned model
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Put model in evaluation mode
        self.model.eval()

    def recognize_text(self, image):
        """
        Recognize text in an image using TrOCR
        
        Args:
            image (PIL.Image): Image to process
            
        Returns:
            str: Recognized text
        """
        # Preprocess image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text