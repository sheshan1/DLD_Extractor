"""
YOLO-based detector for license components
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

from src.utils.config import DEBUG_SAVE_INTERMEDIATES, LICENSE_CLASS_ID, TABLE_CLASS_ID

def preprocess_for_YOLO(image_path):
    """
    Prepares image for YOLO, returns processed image and transformation info.
    
    Args:
        image_path: Path to input image
        
    Returns:
        tuple: (processed image, scale, padding tuple, original PIL image)
    """
    try:
        img_pil = Image.open(image_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
            
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        scale = min(640/w, 640/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Use INTER_LINEAR for resizing, generally good balance
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        processed = np.zeros((640, 640, 3), dtype=np.uint8)
        pad_top = (640 - new_h) // 2
        pad_left = (640 - new_w) // 2
        processed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized
        
        return processed, scale, (pad_top, pad_left), img_pil
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error during preprocess_for_YOLO for {image_path}: {e}")
        return None, None, None, None

def extract_license_components(image_path, yolo_model, debug_dir=None):
    """
    Runs YOLO detection and extracts PIL crops of license and table.
    
    Args:
        image_path: Path to input image
        yolo_model: Loaded YOLO model instance
        debug_dir: Directory to save debug images (optional)
        
    Returns:
        Dictionary with detected components (keys are class IDs)
    """
    processed_img, scale, padding, original_pil = preprocess_for_YOLO(image_path)
    if processed_img is None:
        return None

    original_w, original_h = original_pil.size
    pad_t, pad_l = padding
    components = {}  # Initialize components dictionary

    try:
        results = yolo_model(processed_img, imgsz=640, conf=0.5)
    except Exception as e:
        print(f"Error during YOLO inference for {image_path}: {e}")
        return None  # Return None on inference error

    print(f"  YOLO Found boxes: {len(results[0].boxes) if results and results[0].boxes else 0}")
    
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            try:
                class_id = int(box.cls)
                confidence_scalar = box.conf.item()
                print(f"    Box {i}: Class ID={class_id}, Conf={confidence_scalar:.2f}")
                
                if class_id == LICENSE_CLASS_ID or class_id == TABLE_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    x1_pix, y1_pix = int(x1 * 640), int(y1 * 640)
                    x2_pix, y2_pix = int(x2 * 640), int(y2 * 640)

                    # Check for division by zero if scale is unexpectedly zero
                    if scale == 0:
                        print("Warning: Scale is zero during coordinate adjustment. Skipping box.")
                        continue

                    # Convert from YOLO coordinates back to original image coordinates
                    x1_adj = (x1_pix - pad_l) / scale
                    y1_adj = (y1_pix - pad_t) / scale
                    x2_adj = (x2_pix - pad_l) / scale
                    y2_adj = (y2_pix - pad_t) / scale

                    x1_final = max(0, int(x1_adj))
                    y1_final = max(0, int(y1_adj))
                    x2_final = min(original_w, int(x2_adj))
                    y2_final = min(original_h, int(y2_adj))

                    if x1_final < x2_final and y1_final < y2_final:
                        cropped = original_pil.crop((x1_final, y1_final, x2_final, y2_final))
                        components[class_id] = cropped
                        print(f"      -> Extracted component for Class ID {class_id}")
                        
                        # Save debug images if requested
                        if DEBUG_SAVE_INTERMEDIATES and debug_dir:
                            component_type = "license" if class_id == LICENSE_CLASS_ID else "table"
                            save_path = debug_dir / f"{component_type}_crop.jpg"
                            cropped.save(save_path)
                            print(f"      -> Saved {component_type} crop to {save_path}")
                    else:
                        print(f"Warning: Invalid crop coordinates calculated for class {class_id} "
                              f"(x1:{x1_final}, y1:{y1_final}, x2:{x2_final}, y2:{y2_final}). "
                              f"Skipping.")
            except Exception as box_err:
                print(f"Error processing YOLO box {i}: {box_err}")  # Catch errors processing individual boxes

    if LICENSE_CLASS_ID not in components:
        print("    -> License component NOT found.")
    if TABLE_CLASS_ID not in components:
        print("    -> Table component NOT found.")
        
    return components
