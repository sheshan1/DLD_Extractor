"""
Image preprocessing module for OCR
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.utils.config import DEBUG_SAVE_INTERMEDIATES

def preprocess_image(image_array):
    """
    Applies preprocessing steps to an OpenCV image array.
    
    Args:
        image_array: Input image as numpy array
        
    Returns:
        Preprocessed image as numpy array
    """
        
    img = image_array.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def preprocess_for_ocr(license_crop_pil, table_crop_pil, debug_dir=None):
    """
    Preprocesses images for OCR by checking orientation and preparing table image.
    
    Args:
        license_crop_pil: License component as PIL image (can be None)
        table_crop_pil: Table component as PIL image
        debug_dir: Directory to save debug images (optional)
        
    Returns:
        Table image as numpy array ready for OCR, or None if processing fails
    """
    # Check orientation using license if available
    rotation = 0
    print("  Checking orientation using license component...")
    
    if license_crop_pil:
        try:
            license_crop_cv = cv2.cvtColor(np.array(license_crop_pil), cv2.COLOR_RGB2BGR)
            if license_crop_cv is None or license_crop_cv.size == 0:
                raise ValueError("License crop converted to empty OpenCV array")
                
            osd_img = cv2.cvtColor(license_crop_cv, cv2.COLOR_BGR2GRAY)
            if osd_img is None or osd_img.size == 0:
                raise ValueError("License crop failed grayscale conversion")
                
            _, osd_img_thresh = cv2.threshold(osd_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if osd_img_thresh is None or osd_img_thresh.size == 0:
                raise ValueError("License crop failed thresholding")
                
            osd_config = r'--psm 0 --dpi 300'
            osd_data = pytesseract.image_to_osd(osd_img_thresh, output_type=pytesseract.Output.DICT, config=osd_config)
            rotation = osd_data.get('rotate', 0)
            print(f"    OSD Detected Rotation: {rotation} degrees.")
            
        except pytesseract.TesseractError as ts_err:
            print(f"    Warning: Pytesseract OSD failed: {ts_err}. Assuming 0 rotation.")
            rotation = 0
        except Exception as e:
            print(f"    Warning: Error during OSD pre-processing or execution: {e}. Assuming 0 rotation.")
            rotation = 0
    else:
        print("    Warning: License component not found. Assuming 0 rotation.")
        rotation = 0

    # Rotate table if needed
    rotated_table_pil = table_crop_pil
    if rotation != 0:
        print(f"  Applying {rotation} degree rotation to table crop...")
        try:
            if rotation == 90:
                rotated_table_pil = table_crop_pil.rotate(270, expand=True, fillcolor='white')  # Pytesseract's 90 is clockwise
            elif rotation == 180:
                rotated_table_pil = table_crop_pil.rotate(180, expand=True, fillcolor='white')
            elif rotation == 270:
                rotated_table_pil = table_crop_pil.rotate(90, expand=True, fillcolor='white')  # Pytesseract's 270 is counter-clockwise
            else:
                print(f"    Unusual rotation angle {rotation} detected by OSD. Rotation not applied.")
                rotated_table_pil = table_crop_pil  # Don't rotate on weird angles

            if DEBUG_SAVE_INTERMEDIATES and rotated_table_pil != table_crop_pil and debug_dir:  # Save only if rotation happened
                save_path = debug_dir / "table_rotated.jpg"
                rotated_table_pil.save(save_path)
                print(f"    Saved rotated table to {save_path}")
                
        except Exception as e:
            print(f"    Error rotating table image: {e}. Using unrotated crop.")
            rotated_table_pil = table_crop_pil

    # Convert to OpenCV format for OCR
    try:
        table_image_array = cv2.cvtColor(np.array(rotated_table_pil), cv2.COLOR_RGB2BGR)
        print(f"  Table image shape for OCR: {table_image_array.shape if table_image_array is not None else 'None'}")
        
        if table_image_array is None or table_image_array.size == 0:
            raise ValueError("Converted table array is empty")
            
        return table_image_array
        
    except Exception as e:
        print(f"  Error converting table PIL to OpenCV: {e}")
        return None
