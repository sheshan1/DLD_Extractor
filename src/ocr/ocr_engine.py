"""
OCR module to extract text from images
"""

import cv2
import numpy as np

from src.preprocessing.image_processor import preprocess_image
from src.utils.config import DEBUG_SAVE_INTERMEDIATES

def group_lines(ocr_results, y_tolerance=15):
    """
    Groups PaddleOCR results into lines based on vertical proximity.
    
    Args:
        ocr_results: List of OCR results from PaddleOCR
        y_tolerance: Tolerance for vertical position to consider text on same line
        
    Returns:
        List of structured lines (each line is a list of text items)
    """
    if not ocr_results or not ocr_results[0]:
        return []
        
    items = []
    # PaddleOCR result format: result[0] = [[box, (text, confidence)], ...]
    for line_info in ocr_results[0]:
        try:  # Wrap processing for each line
            if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
                continue
                
            if not isinstance(line_info[0], list) or not isinstance(line_info[1], (list, tuple)) or len(line_info[1]) < 2:
                continue

            box = np.array(line_info[0])
            if box.shape != (4, 2):
                continue

            text, confidence = line_info[1]
            # Ensure confidence is float/int before filtering
            if not isinstance(confidence, (int, float)):
                continue

            center_y = np.mean(box[:, 1])
            center_x = np.mean(box[:, 0])

            if confidence > 0.50:  # Basic confidence filter
                items.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box.tolist(),
                    'center_y': center_y,
                    'center_x': center_x
                })
        except Exception as line_err:
            print(f"Warning: Error processing one OCR line result: {line_err}. Line: {line_info}")  # Warn about specific line errors
            continue  # Skip problematic lines

    if not items:
        return []
        
    items.sort(key=lambda item: item['center_y'])
    lines = []
    current_line = []
    
    if not items:
        return []

    # Grouping logic (wrapped in try-except for safety, though less likely to fail here)
    try:
        current_line.append(items[0])
        line_base_y = items[0]['center_y']
        
        box_np = np.array(items[0]['box'])
        line_avg_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
        
        for i in range(1, len(items)):
            box_np = np.array(items[i]['box'])
            current_item_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
            dynamic_tolerance = max(y_tolerance, (line_avg_height + current_item_height) * 0.4)
            
            if abs(items[i]['center_y'] - line_base_y) <= dynamic_tolerance:
                current_line.append(items[i])
                line_base_y = np.mean([item['center_y'] for item in current_line])
                
                valid_heights = []
                for item in current_line:
                    box_np = np.array(item['box'])
                    h = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 0
                    if h > 0:
                        valid_heights.append(h)
                        
                line_avg_height = np.mean(valid_heights) if valid_heights else 10
            else:
                current_line.sort(key=lambda item: item['center_x'])
                lines.append(current_line)
                current_line = [items[i]]
                line_base_y = items[i]['center_y']
                
                box_np = np.array(items[i]['box'])
                line_avg_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
                
        if current_line:
            current_line.sort(key=lambda item: item['center_x'])
            lines.append(current_line)
            
        structured_lines_text = [[item['text'] for item in line] for line in lines]
        return structured_lines_text
        
    except Exception as group_err:
        print(f"Error during line grouping: {group_err}")
        return []  # Return empty list on error

def run_ocr_pipeline(image_array, ocr_engine, preprocessing_methods=None, debug_prefix="", debug_dir=None):
    """
    Runs preprocessing, OCR, and line grouping on an image array.
    
    Args:
        image_array: Input image as numpy array
        ocr_engine: Initialized OCR engine
        preprocessing_methods: List of preprocessing methods to apply
        debug_prefix: Prefix for debug output messages
        debug_dir: Directory to save debug images
        
    Returns:
        List of structured text lines from the image
    """
    if image_array is None or image_array.size == 0:
        print(f"Warning [{debug_prefix}]: run_ocr received empty image.")
        return []
        
    print(f"  [{debug_prefix}] Running OCR preprocessing...")
    preprocessed = preprocess_image(image_array, preprocessing_methods)
    if preprocessed is None:
        print(f"Warning [{debug_prefix}]: Preprocessing failed.")
        return []

    if DEBUG_SAVE_INTERMEDIATES and debug_dir:
        save_path = debug_dir / "ocr_preprocessed.jpg"
        try:
            cv2.imwrite(str(save_path), preprocessed)
            print(f"    Saved preprocessed image to {save_path}")
        except Exception as e:
            print(f"    Error saving preprocessed image: {e}")

    print(f"  [{debug_prefix}] Running PaddleOCR engine...")
    try:
        result = ocr_engine.ocr(preprocessed, cls=True)
        print(f"    PaddleOCR raw result count: {len(result[0]) if result and result[0] else 0}")
    except Exception as e:
        print(f"Error during PaddleOCR inference [{debug_prefix}]: {e}")
        return []

    print(f"  [{debug_prefix}] Grouping OCR results into lines...")
    structured_lines = group_lines(result, y_tolerance=15)
    print(f"    Grouped into {len(structured_lines)} lines.")
    
    if structured_lines:
        print("      All lines:")
        for i, line in enumerate(structured_lines):
            print(f"        Line {i+1}: {line}")

    return structured_lines
