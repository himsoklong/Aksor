"""
Fine-tuning script for TrOCR model on Khmer text data.
This script fine-tunes the TrOCR model for Khmer text recognition.
"""

import os
import json
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import load_metric

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class KhmerOCRDataset(Dataset):
    """Dataset for fine-tuning TrOCR for Khmer OCR"""
    def __init__(self, root_dir, processor, max_target_length=128):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Load annotations
        with open(os.path.join(root_dir, "annotations.json"), "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
            
        logger.info(f"Loaded dataset with {len(self.annotations)} samples from {root_dir}")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.root_dir, annotation["file_name"])
        text = annotation["text"]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Process text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text  # Store original text for evaluation
        }

def compute_metrics(pred):
    """Compute evaluation metrics: CER (Character Error Rate) and WER (Word Error Rate)"""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 with the pad_token_id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Compute metrics
    cer = load_metric("cer")
    wer = load_metric("wer")
    
    cer_score = cer.compute(predictions=pred_str, references=labels_str)
    wer_score = wer.compute(predictions=pred_str, references=labels_str)
    
    return {
        "cer": cer_score,
        "wer": wer_score
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for Khmer OCR")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="training/data/processed",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="training/output",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--pretrained_model", 
        type=str, 
        default="microsoft/trocr-base-handwritten",
        help="Pretrained TrOCR model to fine-tune"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_target_length", 
        type=int, 
        default=128,
        help="Maximum target sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=2,
        help="Number of gradient accumulation steps"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load processor and model
    global processor
    processor = TrOCRProcessor.from_pretrained(args.pretrained_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
    
    # Set special tokens and resize token embeddings for Khmer language
    # This is important to handle Khmer unicode characters
    processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = args.max_target_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # Resize token embeddings for the new vocabulary
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Create datasets
    train_dataset = KhmerOCRDataset(
        os.path.join(args.data_dir, "train"), 
        processor,
        max_target_length=args.max_target_length
    )
    val_dataset = KhmerOCRDataset(
        os.path.join(args.data_dir, "val"), 
        processor,
        max_target_length=args.max_target_length
    )
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,  # Lower CER is better
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Use fp16 if available
        dataloader_num_workers=4,
        report_to="tensorboard",
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = KhmerOCRDataset(
        os.path.join(args.data_dir, "test"), 
        processor,
        max_target_length=args.max_target_length
    )
    
    results = trainer.evaluate(test_dataset)
    logger.info(f"Test set results: {results}")
    
    # Save test set results
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()