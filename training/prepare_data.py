"""
Data preparation script for TrOCR fine-tuning with Khmer text.
This script processes images and their corresponding text labels to create
a dataset suitable for fine-tuning the TrOCR model.
"""

import os
import json
import shutil
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for TrOCR fine-tuning")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="training/data/raw",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="training/data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.1,
        help="Test split ratio"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def create_directories(output_dir):
    """Create necessary directories for training, validation and test sets"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "train")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "train", "images")).mkdir(exist_ok=True)
    
    Path(os.path.join(output_dir, "val")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "val", "images")).mkdir(exist_ok=True)
    
    Path(os.path.join(output_dir, "test")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "test", "images")).mkdir(exist_ok=True)

def prepare_khmer_ocr_dataset(data_dir, output_dir, val_split, test_split, seed):
    """
    Prepare the dataset for fine-tuning TrOCR for Khmer OCR.
    
    This function:
    1. Reads images and their corresponding text annotations
    2. Splits the data into training, validation, and test sets
    3. Creates the directory structure required for fine-tuning
    4. Copies and processes images to the appropriate directories
    5. Creates annotation files for each split
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Output directory for processed data
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create directory structure
    create_directories(output_dir)
    
    # Read annotations file (expected format: CSV with image_path and text columns)
    annotations_file = os.path.join(data_dir, "annotations.csv")
    if os.path.exists(annotations_file):
        df = pd.read_csv(annotations_file)
        print(f"Loaded {len(df)} annotations from CSV file")
    else:
        # If no annotations file, try to create one from directory structure
        # Assuming data_dir contains image files and corresponding text files
        image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        data = []
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            txt_file = os.path.join(data_dir, f"{base_name}.txt")
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                data.append({
                    'image_path': os.path.join(data_dir, img_file),
                    'text': text
                })
        
        df = pd.DataFrame(data)
        print(f"Created annotations for {len(df)} image-text pairs")
    
    # Split data into train, validation, and test sets
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    test_size = int(len(indices) * test_split)
    val_size = int(len(indices) * val_split)
    train_size = len(indices) - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Split data into {len(train_indices)} training, {len(val_indices)} validation, and {len(test_indices)} test samples")
    
    # Process each split
    splits = [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices)
    ]
    
    for split_name, split_indices in splits:
        split_data = []
        
        for idx in tqdm(split_indices, desc=f"Processing {split_name} split"):
            row = df.iloc[idx]
            img_path = row['image_path']
            text = row['text']
            
            # Get image file name
            img_filename = os.path.basename(img_path)
            new_img_path = os.path.join(output_dir, split_name, "images", img_filename)
            
            # Copy and process image
            try:
                img = Image.open(img_path)
                
                # Basic preprocessing
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(new_img_path)
                
                # Add to annotations
                split_data.append({
                    "file_name": os.path.join("images", img_filename),
                    "text": text
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        # Save annotations as JSON file
        annotations_path = os.path.join(output_dir, split_name, "annotations.json")
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(split_data)} annotations for {split_name} split")

def main():
    args = parse_args()
    prepare_khmer_ocr_dataset(
        args.data_dir,
        args.output_dir,
        args.val_split,
        args.test_split,
        args.seed
    )
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()