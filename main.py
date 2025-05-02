#!/usr/bin/env python3
"""
License OCR Pipeline Main Entry Point

This script orchestrates the license document OCR pipeline:
1. Detects license and table components using YOLO
2. Extracts and preprocesses the components
3. Performs OCR on the table component
4. Extracts and validates vehicle categories and dates
5. Outputs structured data as JSON

Usage:
    python main.py

Configuration is loaded from config.yaml file.
"""

import os
import json
from pathlib import Path

from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.detection.yolo_detector import extract_license_components
from src.preprocessing.image_processor import preprocess_for_ocr
from src.ocr.ocr_engine import run_ocr_pipeline
from src.validation.data_validator import process_ocr_rows
from src.utils.config import (
    YOLO_MODEL_PATH,
    DEBUG_OUTPUT_DIR, 
    DEBUG_SAVE_INTERMEDIATES,
    LICENSE_CLASS_ID,
    TABLE_CLASS_ID,
    VEHICLE_CATEGORIES,
    INPUT_IMAGE_PATH,
    INPUT_FOLDER_PATH,
    OUTPUT_RESULT_FILE,
    OCR_USE_ANGLE_CLS,
    OCR_LANGUAGE,
    OCR_USE_GPU
)

def process_license_image(image_path, yolo_model, ocr_engine):
    """
    Detects components, corrects orientation, runs OCR on table, returns final data.
    
    Args:
        image_path: Path to the input image
        yolo_model: Loaded YOLO model
        ocr_engine: Initialized OCR engine
        
    Returns:
        Dictionary containing extracted data for each vehicle category
    """
    print(f"\n--- Processing Image: {os.path.basename(image_path)} ---")
    base_name = Path(image_path).stem
    
    # Create image-specific debug folder
    debug_img_dir = DEBUG_OUTPUT_DIR / base_name
    if DEBUG_SAVE_INTERMEDIATES:
        debug_img_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Debug intermediates will be saved to: {debug_img_dir}")

    # 1. Run YOLO Detection
    print("  Running YOLO component detection...")
    components = extract_license_components(image_path, yolo_model, debug_img_dir)
    if components is None:
        print("  Error: YOLO component extraction failed.")
        return None
    if not components:
        print("  Warning: YOLO did not detect any license/table components.")
        return None

    license_crop_pil = components.get(LICENSE_CLASS_ID)
    table_crop_pil = components.get(TABLE_CLASS_ID)

    if not table_crop_pil:
        print("  Error: 'Table' component not detected.")
        return None

    # 2. Preprocess table image (includes orientation correction)
    print("  Preprocessing table image...")
    table_image_array = preprocess_for_ocr(license_crop_pil, table_crop_pil, debug_img_dir)
    if table_image_array is None:
        print("  Error: Table image preprocessing failed.")
        return None

    # 3. Run OCR pipeline on the Table Array
    structured_rows = run_ocr_pipeline(
        table_image_array, 
        ocr_engine, 
        debug_prefix=f"{base_name}_table",
        debug_dir=debug_img_dir
    )
    if not structured_rows:
        print("  OCR processing yielded no structured rows.")
        return {}

    # 4. Process OCR Rows (Validation)
    validated_data = process_ocr_rows(structured_rows)

    # 5. Generate Final Output Dictionary
    final_output = {}
    print("\n  --- Final Processed Data ---")
    print(f"  {'Category':<8} | {'Start Date':<12} | {'End Date':<12}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12}")
    for category in VEHICLE_CATEGORIES:
        entry = validated_data.get(category)
        start_date = entry['start_date'] if entry else "--"
        end_date = entry['end_date'] if entry else "--"

        # Only include categories with valid dates
        if start_date != "--" or end_date != "--":
            final_output[category] = {'start_date': start_date, 'end_date': end_date}
            print(f"  {category:<8} | {start_date:<12} | {end_date:<12}")

    return final_output

def save_results_as_json(results, output_path):
    """Save processing results as a JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    """Main entry point for the pipeline, uses configuration from config.yaml"""
    # Validate configuration
    if not INPUT_IMAGE_PATH and not INPUT_FOLDER_PATH:
        print("Error: No input specified in config.yaml. Set either 'input.image_path' or 'input.folder_path'.")
        print("Example configuration:")
        print("input:")
        print("  image_path: 'test_images/example.jpg'  # For a single image")
        print("  # OR")
        print("  folder_path: 'test_images'  # For a folder of images")
        return

    # Create debug output directory
    if DEBUG_SAVE_INTERMEDIATES:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Debug output directory: {DEBUG_OUTPUT_DIR}")

    # Initialize models
    print("Loading YOLO model...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    print("Initializing PaddleOCR engine...")
    try:
        ocr_engine = PaddleOCR(
            use_angle_cls=OCR_USE_ANGLE_CLS,
            lang=OCR_LANGUAGE,
            show_log=False,
            use_gpu=OCR_USE_GPU
        )
    except Exception as e:
        print(f"Error initializing PaddleOCR engine: {e}")
        return

    print("Models loaded successfully.")

    # Process input
    all_results = {}
    
    if INPUT_FOLDER_PATH:
        input_folder = INPUT_FOLDER_PATH
        print(f"\nProcessing all images in folder: {input_folder}")
        # Check if folder exists
        if not os.path.isdir(input_folder):
            print(f"Error: Folder not found at {input_folder}")
            return
            
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
            
        for img_file in image_files:
            image_path = os.path.join(input_folder, img_file)
            final_data = process_license_image(
                image_path, yolo_model, ocr_engine
            )
            # Save results for each image, if null save empty dict
            all_results[img_file] = final_data

    elif INPUT_IMAGE_PATH:
        image_path = INPUT_IMAGE_PATH
        if not os.path.isfile(image_path):
            print(f"Error: Image not found at {image_path}")
            return
            
        print(f"\nProcessing single image: {image_path}")
        final_data = process_license_image(
            image_path, yolo_model, ocr_engine
        )
        # Save results for each image, if null save empty dict
        all_results[os.path.basename(image_path)] = final_data

    # Save results
    save_results_as_json(all_results, OUTPUT_RESULT_FILE)
    print("\n--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()
