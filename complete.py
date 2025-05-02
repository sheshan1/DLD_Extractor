import cv2
import numpy as np
import re
import os
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import pytesseract
from Levenshtein import distance as levenshtein_distance

# --- Constants ---
VEHICLE_CATEGORIES = ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J']
UNCERTAIN_CATEGORY_MARKER = '???'
LICENSE_CLASS_ID = 0 # Assuming 0 is 'license' in your YOLO model
TABLE_CLASS_ID = 1   # Assuming 1 is 'table' in your YOLO model
DEFAULT_OCR_PREPROCESSING = ['gray']
DEBUG_SAVE_INTERMEDIATES = True # Set to True to save intermediate images
DEBUG_OUTPUT_DIR = Path("./debug_output") # Save debug images here

# --- YOLO Helper Functions ---
def preprocess_for_YOLO(image_path):
    """Prepares image for YOLO, returns processed image and transformation info."""
    try:
        img_pil = Image.open(image_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        scale = min(640/w, 640/h)
        new_w, new_h = int(w*scale), int(h*scale)
        # Use INTER_LINEAR for resizing, generally good balance
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        processed = np.zeros((640, 640, 3), dtype=np.uint8)
        pad_top = (640 - new_h) // 2; pad_left = (640 - new_w) // 2
        processed[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized
        return processed, scale, (pad_top, pad_left), img_pil
    except FileNotFoundError: print(f"Error: Image file not found at {image_path}"); return None, None, None, None
    except Exception as e: print(f"Error during preprocess_for_YOLO for {image_path}: {e}"); return None, None, None, None

def extract_license_components(image_path, yolo_model):
    """Runs YOLO detection and extracts PIL crops of license and table."""
    processed_img, scale, padding, original_pil = preprocess_for_YOLO(image_path)
    if processed_img is None: return None

    original_w, original_h = original_pil.size
    pad_t, pad_l = padding
    components = {} # Initialize components dictionary

    try:
        results = yolo_model(processed_img, imgsz=640, conf=0.5)
    except Exception as e:
        print(f"Error during YOLO inference for {image_path}: {e}")
        return None # Return None on inference error

    print(f"  YOLO Found boxes: {len(results[0].boxes) if results and results[0].boxes else 0}")
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            try:
                class_id = int(box.cls)
                # --- FIX: Convert tensor to scalar using .item() before formatting ---
                confidence_scalar = box.conf.item()
                print(f"    Box {i}: Class ID={class_id}, Conf={confidence_scalar:.2f}")
                # -----------------------------------------------------------------------
                if class_id == LICENSE_CLASS_ID or class_id == TABLE_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    x1_pix, y1_pix = int(x1 * 640), int(y1 * 640)
                    x2_pix, y2_pix = int(x2 * 640), int(y2 * 640)

                    # Check for division by zero if scale is unexpectedly zero
                    if scale == 0: print("Warning: Scale is zero during coordinate adjustment. Skipping box."); continue

                    x1_adj = (x1_pix - pad_l) / scale; y1_adj = (y1_pix - pad_t) / scale
                    x2_adj = (x2_pix - pad_l) / scale; y2_adj = (y2_pix - pad_t) / scale

                    x1_final = max(0, int(x1_adj)); y1_final = max(0, int(y1_adj))
                    x2_final = min(original_w, int(x2_adj)); y2_final = min(original_h, int(y2_adj))

                    if x1_final < x2_final and y1_final < y2_final:
                        cropped = original_pil.crop((x1_final, y1_final, x2_final, y2_final))
                        components[class_id] = cropped
                        print(f"      -> Extracted component for Class ID {class_id}")
                    else: print(f"Warning: Invalid crop coordinates calculated for class {class_id} (x1:{x1_final}, y1:{y1_final}, x2:{x2_final}, y2:{y2_final}). Skipping.")
            except Exception as box_err:
                print(f"Error processing YOLO box {i}: {box_err}") # Catch errors processing individual boxes

    if LICENSE_CLASS_ID not in components: print("    -> License component NOT found.")
    if TABLE_CLASS_ID not in components: print("    -> Table component NOT found.")
    return components


# --- OCR Helper Functions ---
def preprocess_image(image_array, methods=None):
    """Applies preprocessing steps to an OpenCV image array."""
    if methods is None: methods = DEFAULT_OCR_PREPROCESSING
    if image_array is None or image_array.size == 0: print("Warning: preprocess_image received empty array."); return None
    img = image_array.copy()

    try: # Wrap core processing in try-except
        is_gray = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        original_was_color = len(image_array.shape) == 3 and not is_gray

        if 'gray' in methods and not is_gray:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); img = img_gray if img_gray is not None else img; is_gray = True
        if 'median_blur' in methods and img is not None: img = cv2.medianBlur(img, 3)
        if 'gaussian_blur' in methods and img is not None: img = cv2.GaussianBlur(img, (5, 5), 0)
        if 'denoise' in methods and img is not None:
            if original_was_color and not is_gray: # Need color input for fastNlMeansDenoisingColored
                 img_denoise_input = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert back if needed
                 if img_denoise_input is not None: img = cv2.fastNlMeansDenoisingColored(img_denoise_input, None, 10, 10, 7, 21); is_gray = False
            elif is_gray: img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        if 'hist_eq' in methods and is_gray and img is not None: img = cv2.equalizeHist(img)
        if 'clahe' in methods and is_gray and img is not None: clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); img = clahe.apply(img)
        if 'sharpen' in methods and img is not None: kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]); img = cv2.filter2D(img, -1, kernel)
        thresholded = False
        if 'multi_threshold' in methods and is_gray and img is not None: _, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU); th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2); img = cv2.bitwise_and(th1, th2); thresholded = True
        if 'adaptive_threshold' in methods and not thresholded and is_gray and img is not None: img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if img is None or img.size == 0: print("Warning: Preprocessing resulted in empty image."); return None
        return img
    except cv2.error as cv_err:
        print(f"OpenCV error during preprocessing: {cv_err}")
        return None # Return None on OpenCV errors
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")
        return None

def group_lines(ocr_results, y_tolerance=15):
    """Groups PaddleOCR results into lines based on vertical proximity."""
    if not ocr_results or not ocr_results[0]: return []
    items = []
    # PaddleOCR result format: result[0] = [[box, (text, confidence)], ...]
    for line_info in ocr_results[0]:
        try: # Wrap processing for each line
            if not isinstance(line_info, (list, tuple)) or len(line_info) < 2: continue
            if not isinstance(line_info[0], list) or not isinstance(line_info[1], (list, tuple)) or len(line_info[1]) < 2: continue

            box = np.array(line_info[0])
            if box.shape != (4, 2): continue

            text, confidence = line_info[1]
            # Ensure confidence is float/int before filtering
            if not isinstance(confidence, (int, float)): continue

            center_y = np.mean(box[:, 1]); center_x = np.mean(box[:, 0])

            if confidence > 0.50: # Basic confidence filter
                 items.append({'text': text, 'confidence': confidence, 'box': box.tolist(), 'center_y': center_y, 'center_x': center_x})
        except Exception as line_err:
             print(f"Warning: Error processing one OCR line result: {line_err}. Line: {line_info}") # Warn about specific line errors
             continue # Skip problematic lines

    if not items: return []
    items.sort(key=lambda item: item['center_y'])
    lines = []; current_line = []
    if not items: return []

    # Grouping logic (wrapped in try-except for safety, though less likely to fail here)
    try:
        current_line.append(items[0]); line_base_y = items[0]['center_y']
        box_np = np.array(items[0]['box']); line_avg_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
        for i in range(1, len(items)):
            box_np = np.array(items[i]['box']); current_item_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
            dynamic_tolerance = max(y_tolerance, (line_avg_height + current_item_height) * 0.4)
            if abs(items[i]['center_y'] - line_base_y) <= dynamic_tolerance:
                current_line.append(items[i])
                line_base_y = np.mean([item['center_y'] for item in current_line])
                valid_heights = []
                for item in current_line:
                    box_np = np.array(item['box']); h = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 0
                    if h > 0: valid_heights.append(h)
                line_avg_height = np.mean(valid_heights) if valid_heights else 10
            else:
                current_line.sort(key=lambda item: item['center_x']); lines.append(current_line)
                current_line = [items[i]]; line_base_y = items[i]['center_y']
                box_np = np.array(items[i]['box']); line_avg_height = (np.max(box_np[:, 1]) - np.min(box_np[:, 1])) if box_np.size > 0 else 10
        if current_line: current_line.sort(key=lambda item: item['center_x']); lines.append(current_line)
        structured_lines_text = [[item['text'] for item in line] for line in lines]
        return structured_lines_text
    except Exception as group_err:
        print(f"Error during line grouping: {group_err}")
        return [] # Return empty list on error

def add_years_to_date(source_date, years_to_add):
    """ Safely adds years to a date object, handling leap years. """
    try:
        new_year = source_date.year + years_to_add; return source_date.replace(year=new_year)
    except ValueError:
        if source_date.month == 2 and source_date.day == 29:
            try: return date(new_year, 2, 28)
            except ValueError: return None
        else: return None

def validate_and_format_date(text):
    """ Validates text against various date patterns, returns 'DD.MM.YYYY' or None. """
    if not isinstance(text, str): return None
    cleaned_text = text.strip().replace(' ', '').replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('!', '1').replace('S', '5').replace('s', '5').replace('B', '8').replace('g', '9').replace('q','9')
    patterns = [
        {'regex': r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', 'groups': (1, 2, 3)}, {'regex': r'^(\d{1,2})\.(\d{1,2})(\d{4})$', 'groups': (1, 2, 3)},
        {'regex': r'^(\d{1,2})(\d{1,2})\.(\d{4})$', 'groups': (1, 2, 3)}, {'regex': r'^(\d{4})\.(\d{1,2})\.(\d{1,2})$', 'groups': (3, 2, 1)},
        {'regex': r'^(\d{2})(\d{2})(\d{4})$', 'groups': (1, 2, 3)}, {'regex': r'^(\d{4})(\d{2})(\d{2})$', 'groups': (3, 2, 1)},
        {'regex': r'^(\d{2})(\d{2})(\d{2})$', 'groups': (1, 2, 3), 'infer_century': True},
    ]
    for p in patterns:
        match = re.fullmatch(p['regex'], cleaned_text)
        if match:
            try:
                day_idx, month_idx, year_idx = p['groups']; dd_str = match.group(day_idx); mm_str = match.group(month_idx); yyyy_str = match.group(year_idx)
                if p.get('infer_century', False): year = int(yyyy_str); current_year_short = date.today().year % 100; yyyy_str = f"20{yyyy_str.zfill(2)}" if year <= current_year_short + 10 else f"19{yyyy_str.zfill(2)}"
                dd_str = dd_str.zfill(2); mm_str = mm_str.zfill(2)
                if len(yyyy_str) != 4: continue
                dt_obj = datetime(int(yyyy_str), int(mm_str), int(dd_str)); return dt_obj.strftime('%d.%m.%Y')
            except (ValueError, IndexError): continue
    return None

def run_ocr_on_image_array(image_array, ocr_engine, preprocessing_methods=None, debug_prefix=""):
    """Runs preprocessing, OCR, and line grouping on an image array."""
    if image_array is None or image_array.size == 0: print(f"Warning [{debug_prefix}]: run_ocr received empty image."); return []
    print(f"  [{debug_prefix}] Running OCR preprocessing...")
    preprocessed = preprocess_image(image_array, preprocessing_methods)
    if preprocessed is None: print(f"Warning [{debug_prefix}]: Preprocessing failed."); return []

    if DEBUG_SAVE_INTERMEDIATES:
        save_path = DEBUG_OUTPUT_DIR / f"{debug_prefix}_3_ocr_preprocessed.jpg"
        try: cv2.imwrite(str(save_path), preprocessed); print(f"    Saved preprocessed image to {save_path}")
        except Exception as e: print(f"    Error saving preprocessed image: {e}")

    print(f"  [{debug_prefix}] Running PaddleOCR engine...")
    try:
        result = ocr_engine.ocr(preprocessed, cls=True)
        print(f"    PaddleOCR raw result count: {len(result[0]) if result and result[0] else 0}")
    except Exception as e: print(f"Error during PaddleOCR inference [{debug_prefix}]: {e}"); return []

    print(f"  [{debug_prefix}] Grouping OCR results into lines...")
    structured_lines = group_lines(result, y_tolerance=15) # Use group_lines now
    print(f"    Grouped into {len(structured_lines)} lines.")
    if structured_lines:
        print("      All lines:")
        for i, line in enumerate(structured_lines): print(f"        Line {i+1}: {line}")

    return structured_lines


# --- [Keep previous imports and other functions] ---

def process_ocr_rows(structured_rows):
    """
    Applies category and date validation rules to structured OCR rows.
    [REVISED: Includes step to infer missing categories based on order]
    """
    # Store results per row initially, including failures
    row_processing_results = []
    all_potential_dates_flat = [] # Keep flat list for frequency analysis
    today = date.today()

    print(f"  Processing {len(structured_rows)} structured rows for initial validation...")

    # --- Step 1: Initial Category Validation & Date Extraction per Row ---
    for row_index, row in enumerate(structured_rows):
        if not row:
            row_processing_results.append({
                'row_index': row_index, 'original_row': row,
                'initial_category': UNCERTAIN_CATEGORY_MARKER,
                'potential_dates': []
            })
            continue

        initial_category = UNCERTAIN_CATEGORY_MARKER
        potential_dates_in_row = []
        category_text_raw = "" # Store the raw text OCR'd as category

        try:
            category_text_raw = row[0].strip()
            cleaned_category_text = re.sub(r'[.,;:!?-]$', '', category_text_raw).upper().replace(' ','')
            # --- Basic OCR corrections for categories ---
            cleaned_category_text = cleaned_category_text.replace('8', 'B').replace('1', 'I') # Common swaps
            # -------------------------------------------

            if not cleaned_category_text: # Handle cases where first element is empty after cleaning
                print(f"      Row {row_index+1}: Empty category text after cleaning.")
                initial_category = UNCERTAIN_CATEGORY_MARKER
            elif cleaned_category_text in VEHICLE_CATEGORIES:
                initial_category = cleaned_category_text
                print(f"      Row {row_index+1}: Valid category '{initial_category}' found directly.")
            else:
                # Try Levenshtein distance
                min_dist = float('inf')
                closest_category = None
                for known_cat in VEHICLE_CATEGORIES:
                    dist = levenshtein_distance(cleaned_category_text, known_cat)
                    if dist < min_dist:
                        min_dist = dist
                        closest_category = known_cat
                # Allow distance 1, or maybe 0 if strictness needed
                if min_dist <= 1:
                    initial_category = closest_category
                    print(f"      Row {row_index+1}: Category '{closest_category}' matched via Levenshtein (Dist={min_dist}) from '{cleaned_category_text}'.")
                else:
                    print(f"      Row {row_index+1}: Category text '{cleaned_category_text}' (Raw: '{category_text_raw}') not matched. Marked as '???'.")
                    initial_category = UNCERTAIN_CATEGORY_MARKER

            # Date Extraction for this row (regardless of category success initially)
            for col_index, text in enumerate(row):
                 # Skip if it was the (potentially misidentified) category column
                 if col_index == 0 and initial_category != UNCERTAIN_CATEGORY_MARKER : continue
                 # If category was '???', check if col 0 IS a date
                 if col_index == 0 and initial_category == UNCERTAIN_CATEGORY_MARKER and not validate_and_format_date(text): continue

                 formatted_date_str = validate_and_format_date(text)
                 if formatted_date_str:
                     try:
                         date_obj = datetime.strptime(formatted_date_str, '%d.%m.%Y').date()
                         if 1950 <= date_obj.year <= today.year + 30:
                            date_info = {'dt': date_obj, 'str': formatted_date_str, 'row_idx': row_index}
                            potential_dates_in_row.append(date_info)
                            all_potential_dates_flat.append(date_info) # Add to flat list
                     except ValueError: continue

            print(f"        -> Extracted {len(potential_dates_in_row)} potential dates from row {row_index+1}.")

        except IndexError:
            print(f"Warning: IndexError processing row {row_index+1}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER # Mark as uncertain
        except Exception as row_err:
            print(f"Warning: Error processing row {row_index+1}: {row_err}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER # Mark as uncertain

        row_processing_results.append({
            'row_index': row_index, 'original_row': row,
            'initial_category': initial_category,
            'potential_dates': potential_dates_in_row
        })

    # --- Step 2: Infer Missing Categories Based on Order ---
    print("\n  Attempting to infer missing categories based on row order...")
    num_inferred = 0
    for i in range(len(row_processing_results)):
        current_result = row_processing_results[i]

        # Check if candidate for inference: Category is '???' but dates exist
        if current_result['initial_category'] == UNCERTAIN_CATEGORY_MARKER and current_result['potential_dates']:
            print(f"    Row {i+1}: Candidate for missing category inference (Category='???', Dates found).")
            try:
                # Check neighbors (ensure they exist and have valid categories)
                if i > 0 and i < len(row_processing_results) - 1:
                    prev_result = row_processing_results[i-1]
                    next_result = row_processing_results[i+1]

                    prev_cat = prev_result['initial_category']
                    next_cat = next_result['initial_category']

                    # Neighbors must be valid categories
                    if prev_cat != UNCERTAIN_CATEGORY_MARKER and next_cat != UNCERTAIN_CATEGORY_MARKER:
                        print(f"      -> Neighbors: Prev='{prev_cat}' (Row {i}), Next='{next_cat}' (Row {i+2}).")
                        # Check order in master list
                        try:
                            prev_idx = VEHICLE_CATEGORIES.index(prev_cat)
                            next_idx = VEHICLE_CATEGORIES.index(next_cat)

                            if next_idx == prev_idx + 2:
                                inferred_category = VEHICLE_CATEGORIES[prev_idx + 1]
                                print(f"      -> SUCCESS: Inferred missing category as '{inferred_category}'. Updating Row {i+1}.")
                                current_result['initial_category'] = inferred_category
                                num_inferred += 1
                            else:
                                print(f"      -> Order check failed: Indices {prev_idx} and {next_idx} are not separated by 1.")
                        except ValueError:
                             print(f"      -> Error: Neighbor category not found in VEHICLE_CATEGORIES list.")
                    else:
                         print(f"      -> One or both neighbors have uncertain categories.")
                else:
                     print(f"      -> Cannot check neighbors (first or last row).")
            except Exception as infer_err:
                print(f"      -> Error during inference check for row {i+1}: {infer_err}")

    print(f"    Inferred {num_inferred} missing categories.")

    # --- Step 3: Aggregate Data for Validated Categories ---
    category_potential_dates = {} # Rebuild this based on potentially updated results
    all_potential_dates = []      # Rebuild this flat list too

    print("\n  Aggregating dates for validated/inferred categories...")
    for result in row_processing_results:
        final_category = result['initial_category']
        if final_category != UNCERTAIN_CATEGORY_MARKER:
            if final_category not in category_potential_dates:
                category_potential_dates[final_category] = []
            # Add dates from this row to the category's list and the flat list
            for date_info in result['potential_dates']:
                # Prevent duplicates if a category was somehow processed twice (shouldn't happen with this structure)
                if date_info not in category_potential_dates[final_category]:
                     category_potential_dates[final_category].append(date_info)
                # Add to flat list (duplicates ok here, handled by frequency count)
                all_potential_dates.append(date_info) # Use the validated flat list
            print(f"    Added {len(result['potential_dates'])} dates for category '{final_category}' (from row {result['row_index']+1})")


    # --- Step 4: Analyze Frequencies ---
    # Use the 'all_potential_dates' list built in Step 3
    date_str_counts = {}
    for d_info in all_potential_dates:
        date_str_counts[d_info['str']] = date_str_counts.get(d_info['str'], 0) + 1

    print("\n  Analyzing date frequencies (post-inference)...")
    if not all_potential_dates:
         print("    No potential dates found after category validation/inference."); return {}
    if date_str_counts:
        sorted_date_counts = sorted(date_str_counts.items(), key=lambda item: item[1], reverse=True)
        print("    Detected date frequencies:")
        for d_str, count in sorted_date_counts: print(f"      - '{d_str}': {count} time(s)")
    else: print("    No dates found to analyze frequency for."); return {} # Should not happen if all_potential_dates is not empty

    # Prepare global lists needed for date pairing inference
    all_valid_start_dates_info = sorted([d for d in all_potential_dates if d['dt'] <= today], key=lambda x: x['dt'])
    print(f"    All Valid Start Dates (<= today): {[d['str'] for d in all_valid_start_dates_info]}")
    print(f"    All Potential Dates (for end check): {[d['str'] for d in sorted(all_potential_dates, key=lambda x: x['dt'])]}")


    # --- Step 5: Date Pairing (Rule 1) & Single Candidate Identification (Rule 2) ---
    # This now uses the 'category_potential_dates' populated in Step 3
    final_validated_data = {} # Store confirmed pairs here
    single_candidates = {} # Store {category: {'known_part': 'start'/'end', 'info': date_info_dict}}
    processed_categories_pairing = set() # Use a new set for this stage

    print("\n  Assigning initial date pairs and identifying single candidates...")
    # Use the aggregated category_potential_dates dictionary
    for category, potential_dates in category_potential_dates.items():
        # Category here is guaranteed to be valid (not '???')
        if category in processed_categories_pairing: continue

        print(f"    Processing Category for Date Pairing: {category}")
        potential_dates.sort(key=lambda x: x['dt'])

        # --- Rule 1: Find direct pair ---
        best_direct_pair = None
        possible_pairs = []
        for i in range(len(potential_dates)):
            d1 = potential_dates[i]
            if d1['dt'] > today: continue
            for j in range(len(potential_dates)):
                if i == j: continue
                d2 = potential_dates[j]
                if d1['dt'] < d2['dt']:
                    freq_score = date_str_counts.get(d1['str'], 0) + date_str_counts.get(d2['str'], 0)
                    possible_pairs.append({'start': d1, 'end': d2, 'score': freq_score})

        if possible_pairs:
            best_direct_pair = max(possible_pairs, key=lambda p: p['score'])
            print(f"      -> Rule 1 MATCH: Found direct pair {best_direct_pair['start']['str']} -> {best_direct_pair['end']['str']}")
            final_validated_data[category] = {
                'start_date': best_direct_pair['start']['str'],
                'end_date': best_direct_pair['end']['str']
            }
            processed_categories_pairing.add(category)
            continue

        # --- Rule 2: Identify single candidates ---
        category_starts = [d for d in potential_dates if d['dt'] <= today]
        category_potential_ends = [d for d in potential_dates if d not in category_starts]

        if len(category_starts) == 1 and len(category_potential_ends) == 0:
            single_candidates[category] = {'known_part': 'start', 'info': category_starts[0]}
            print(f"      -> Rule 2: Found single start candidate: {category_starts[0]['str']}")
            processed_categories_pairing.add(category)
        elif len(category_starts) == 0 and len(category_potential_ends) == 1:
            single_candidates[category] = {'known_part': 'end', 'info': category_potential_ends[0]}
            print(f"      -> Rule 2: Found single end candidate: {category_potential_ends[0]['str']}")
            processed_categories_pairing.add(category)
        elif len(category_starts) == 1 and len(category_potential_ends) == 1:
             if category_starts[0]['dt'] < category_potential_ends[0]['dt']:
                 print(f"      -> Rule 2 WARNING: Found 1 start ({category_starts[0]['str']}) and 1 potential end ({category_potential_ends[0]['str']}) but Rule 1 failed? Storing pair.")
                 final_validated_data[category] = {
                     'start_date': category_starts[0]['str'],
                     'end_date': category_potential_ends[0]['str']
                 }
             else:
                single_candidates[category] = {'known_part': 'start', 'info': category_starts[0]}
                print(f"      -> Rule 2 (Non-chronological): Treating as single start candidate: {category_starts[0]['str']}")
             processed_categories_pairing.add(category)
        else:
            print(f"      -> No direct pair and zero or multiple/ambiguous start/end dates for {category}. Cannot infer dates.")
            processed_categories_pairing.add(category)

    # --- Step 6: Date Inference Pass - Complete Pairs for Single Candidates ---
    # [This logic remains the same as the previous version]
    print("\n  Date Inference Pass: Attempting to complete pairs...")
    DD_MM_BONUS_SCORE = 10 # Weight for matching day/month

    for category, candidate_data in single_candidates.items():
        if category in final_validated_data: continue # Skip if already assigned in Rule 2 warning case

        known_part_type = candidate_data['known_part']
        known_info = candidate_data['info']
        known_dt = known_info['dt']
        known_str = known_info['str']

        print(f"    Inferring date partner for {category} (Known {known_part_type}: {known_str})")
        best_partner_info = None
        highest_inference_score = -1

        if known_part_type == 'start':
            potential_partners = [p for p in all_potential_dates if p['dt'] > known_dt]
            # print(f"      Potential End Partners: {[p['str'] for p in sorted(potential_partners, key=lambda x: x['dt'])]}") # Verbose
            for partner_info in potential_partners:
                partner_str = partner_info['str']; partner_dt = partner_info['dt']
                score = date_str_counts.get(partner_str, 0)
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month: score += DD_MM_BONUS_SCORE
                if score > highest_inference_score:
                    highest_inference_score = score; best_partner_info = partner_info
                elif score == highest_inference_score and best_partner_info and partner_dt < best_partner_info['dt']: best_partner_info = partner_info

            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best End Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                final_validated_data[category] = {'start_date': known_str, 'end_date': best_partner_info['str']}
            else:
                 print(f"      -> DATE INFERENCE FAILED: No suitable end partner found.")
                 final_validated_data.setdefault(category, {'start_date': "--", 'end_date': "--"})

        elif known_part_type == 'end':
            potential_partners = [p for p in all_valid_start_dates_info if p['dt'] < known_dt]
            # print(f"      Potential Start Partners: {[p['str'] for p in reversed(sorted(potential_partners, key=lambda x: x['dt']))]}") # Verbose
            for partner_info in potential_partners:
                partner_str = partner_info['str']; partner_dt = partner_info['dt']
                score = date_str_counts.get(partner_str, 0)
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month: score += DD_MM_BONUS_SCORE
                if score > highest_inference_score:
                    highest_inference_score = score; best_partner_info = partner_info
                elif score == highest_inference_score and best_partner_info and partner_dt > best_partner_info['dt']: best_partner_info = partner_info

            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best Start Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                final_validated_data[category] = {'start_date': best_partner_info['str'], 'end_date': known_str}
            else:
                print(f"      -> DATE INFERENCE FAILED: No suitable start partner found.")
                final_validated_data.setdefault(category, {'start_date': "--", 'end_date': "--"})

    # --- Final Output Generation ---
    # [This logic remains the same]
    print("\n  --- Final Processed Data (Category & Date Inferred Pairs Included) ---")
    print(f"  {'Category':<8} | {'Start Date':<12} | {'End Date':<12}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12}")
    final_output_dict = {}
    for category in VEHICLE_CATEGORIES:
        entry = final_validated_data.get(category, {'start_date': "--", 'end_date': "--"})
        start_date = entry['start_date']
        end_date = entry['end_date']
        final_output_dict[category] = {'start_date': start_date, 'end_date': end_date}
        print(f"  {category:<8} | {start_date:<12} | {end_date:<12}")

    return final_output_dict

# --- Main Orchestration Function ---
def process_license_image(image_path, yolo_model, ocr_engine, ocr_preprocessing_methods=None):
    """ Detects components, corrects orientation, runs OCR on table, returns final data. """
    print(f"\n--- Processing Image: {os.path.basename(image_path)} ---")
    base_name = Path(image_path).stem # Get filename without ext for debug saving
    if DEBUG_SAVE_INTERMEDIATES:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Debug intermediates will be saved to: {DEBUG_OUTPUT_DIR}")

    # 1. Run YOLO Detection
    print("  Running YOLO component detection...")
    components = extract_license_components(image_path, yolo_model)
    if components is None: print("  Error: YOLO component extraction failed."); return None # Check for None explicitly
    if not components: print("  Warning: YOLO did not detect any license/table components."); return None # Handle empty dict

    license_crop_pil = components.get(LICENSE_CLASS_ID)
    table_crop_pil = components.get(TABLE_CLASS_ID)

    if DEBUG_SAVE_INTERMEDIATES:
        if license_crop_pil:
            try: license_crop_pil.save(DEBUG_OUTPUT_DIR / f"{base_name}_0_license_crop.jpg")
            except Exception as e: print(f"    Error saving license crop: {e}")
        if table_crop_pil:
            try: table_crop_pil.save(DEBUG_OUTPUT_DIR / f"{base_name}_1_table_crop.jpg")
            except Exception as e: print(f"    Error saving table crop: {e}")

    if not table_crop_pil: print("  Error: 'Table' component not detected."); return None

    # 2. Orientation Check using License
    rotation = 0
    print("  Checking orientation using license component...")
    if license_crop_pil:
        try:
            license_crop_cv = cv2.cvtColor(np.array(license_crop_pil), cv2.COLOR_RGB2BGR)
            if license_crop_cv is None or license_crop_cv.size == 0: raise ValueError("License crop converted to empty OpenCV array")
            osd_img = cv2.cvtColor(license_crop_cv, cv2.COLOR_BGR2GRAY)
            if osd_img is None or osd_img.size == 0: raise ValueError("License crop failed grayscale conversion")
            _, osd_img_thresh = cv2.threshold(osd_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if osd_img_thresh is None or osd_img_thresh.size == 0: raise ValueError("License crop failed thresholding")
            osd_config = r'--psm 0 --dpi 300'
            osd_data = pytesseract.image_to_osd(osd_img_thresh, output_type=pytesseract.Output.DICT, config=osd_config)
            rotation = osd_data.get('rotate', 0)
            print(f"    OSD Detected Rotation: {rotation} degrees.")
        except pytesseract.TesseractError as ts_err: print(f"    Warning: Pytesseract OSD failed: {ts_err}. Assuming 0 rotation."); rotation = 0
        except Exception as e: print(f"    Warning: Error during OSD pre-processing or execution: {e}. Assuming 0 rotation."); rotation = 0
    else: print("    Warning: License component not found. Assuming 0 rotation."); rotation = 0

    # 3. Rotate Table Crop if needed
    rotated_table_pil = table_crop_pil
    if rotation != 0:
        print(f"  Applying {rotation} degree rotation to table crop...")
        try:
            if rotation == 90: rotated_table_pil = table_crop_pil.rotate(270, expand=True, fillcolor='white') # Pytesseract's 90 is clockwise
            elif rotation == 180: rotated_table_pil = table_crop_pil.rotate(180, expand=True, fillcolor='white')
            elif rotation == 270: rotated_table_pil = table_crop_pil.rotate(90, expand=True, fillcolor='white') # Pytesseract's 270 is counter-clockwise
            else: print(f"    Unusual rotation angle {rotation} detected by OSD. Rotation not applied."); rotated_table_pil = table_crop_pil # Don't rotate on weird angles

            if DEBUG_SAVE_INTERMEDIATES and rotated_table_pil != table_crop_pil: # Save only if rotation happened
                 rotated_table_pil.save(DEBUG_OUTPUT_DIR / f"{base_name}_2_table_rotated.jpg")
        except Exception as e: print(f"    Error rotating table image: {e}. Using unrotated crop."); rotated_table_pil = table_crop_pil

    # 4. Prepare Table Image Array for OCR
    try:
        table_image_array = cv2.cvtColor(np.array(rotated_table_pil), cv2.COLOR_RGB2BGR)
        print(f"  Table image shape for OCR: {table_image_array.shape if table_image_array is not None else 'None'}")
        if table_image_array is None or table_image_array.size == 0: raise ValueError("Converted table array is empty")
    except Exception as e: print(f"  Error converting table PIL to OpenCV: {e}"); return None

    # 5. Run OCR pipeline on the Table Array
    structured_rows = run_ocr_on_image_array(table_image_array, ocr_engine, ocr_preprocessing_methods, debug_prefix=f"{base_name}_table")
    if not structured_rows: print("  OCR processing yielded no structured rows.")

    # 6. Process OCR Rows (Validation)
    validated_data = process_ocr_rows(structured_rows)

    # 7. Generate Final Output Dictionary
    final_output = {}
    print("\n  --- Final Processed Data ---")
    print(f"  {'Category':<8} | {'Start Date':<12} | {'End Date':<12}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12}")
    for category in VEHICLE_CATEGORIES:
        entry = validated_data.get(category)
        start_date = entry['start_date'] if entry else "--"
        end_date = entry['end_date'] if entry else "--"
        final_output[category] = {'start_date': start_date, 'end_date': end_date}
        print(f"  {category:<8} | {start_date:<12} | {end_date:<12}")

    return final_output

if __name__ == "__main__":
    # --- Configuration ---
    YOLO_MODEL_PATH = '/home/sheshan/Desktop/Projects/NewTry/best.pt' # !!! UPDATE THIS PATH !!!
    # --- CHOOSE INPUT ---
    # INPUT_IMAGE_PATH = "/kaggle/input/testing/two.jpeg"
    # INPUT_IMAGE_PATH = "/kaggle/input/testing/1_0_rotated.jpg"
    INPUT_IMAGE_PATH = "/home/sheshan/Desktop/Projects/NewTry/test_images/170349.jpg"
    INPUT_FOLDER_PATH = None # Set to a folder path to process multiple images

    # --- Optional: Kaggle Tesseract Setup ---
    # Run this in a separate cell if needed:
    # !sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
    # !pip install python-Levenshtein pytesseract ultralytics --quiet

    # Check if YOLO model exists
    if not Path(YOLO_MODEL_PATH).is_file():
        print(f"FATAL ERROR: YOLO model not found at '{YOLO_MODEL_PATH}'. Please update the path."); exit()

    # --- Initialize Models ---
    print("Loading YOLO model...")
    try: yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e: print(f"Error loading YOLO model: {e}"); exit()
    print("Initializing PaddleOCR engine...")
    try: ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False) # Specify use_gpu if applicable
    except Exception as e: print(f"Error initializing PaddleOCR engine: {e}"); exit()
    print("Models loaded successfully.")

    # --- Process Input ---
    all_results = {}
    if INPUT_FOLDER_PATH and os.path.isdir(INPUT_FOLDER_PATH):
        print(f"\nProcessing all images in folder: {INPUT_FOLDER_PATH}")
        image_files = [f for f in os.listdir(INPUT_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not image_files: print(f"No image files found in {INPUT_FOLDER_PATH}")
        for img_file in image_files:
            image_path = os.path.join(INPUT_FOLDER_PATH, img_file)
            final_data = process_license_image(image_path, yolo_model, ocr_engine, DEFAULT_OCR_PREPROCESSING)
            all_results[img_file] = final_data
    elif INPUT_IMAGE_PATH and os.path.isfile(INPUT_IMAGE_PATH):
        print(f"\nProcessing single image: {INPUT_IMAGE_PATH}")
        final_data = process_license_image(INPUT_IMAGE_PATH, yolo_model, ocr_engine, DEFAULT_OCR_PREPROCESSING)
        all_results[os.path.basename(INPUT_IMAGE_PATH)] = final_data
    else: print("Error: No valid input image or folder specified.")

    print("\n--- Script Finished ---")