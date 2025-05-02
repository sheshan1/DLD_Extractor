"""
Data validation module for OCR results

This module analyzes OCR-extracted text from vehicle licensing documents to:
1. Identify and validate vehicle category codes (A, B, C, etc.)
2. Extract and validate dates (start and end dates for each category)
3. Pair start dates with corresponding end dates
4. Infer missing information using contextual patterns
"""

import re
from datetime import datetime, date
from Levenshtein import distance as levenshtein_distance

from src.utils.helpers import validate_and_format_date
from src.utils.config import VEHICLE_CATEGORIES, UNCERTAIN_CATEGORY_MARKER


def _validate_category(category_text_raw):
    """
    Validate and attempt to correct category text from OCR.
    
    Args:
        category_text_raw: Raw text from OCR for potential category
        
    Returns:
        tuple: (validated_category, description_for_logging)
    """
    # Handle empty input
    if not category_text_raw:
        return UNCERTAIN_CATEGORY_MARKER, "Empty category text"
    
    # Clean and normalize text - remove punctuation, convert to uppercase, remove spaces
    cleaned_category_text = re.sub(r'[.,;:!?-]$', '', category_text_raw).upper().replace(' ', '')
    
    # Apply common OCR correction patterns for vehicle categories
    # e.g., '8' is often misread as 'B', 'I' as '1'
    cleaned_category_text = cleaned_category_text.replace('8', 'B').replace('I', '1')
    
    # Empty after cleaning - happens when input was just punctuation/spaces
    if not cleaned_category_text:
        return UNCERTAIN_CATEGORY_MARKER, "Empty category text after cleaning"
    # Direct match to known categories - most confident case
    elif cleaned_category_text in VEHICLE_CATEGORIES:
        return cleaned_category_text, f"Valid category '{cleaned_category_text}' found directly"
    
    # Try fuzzy matching using Levenshtein distance for OCR errors
    min_dist = float('inf')
    closest_category = None
    for known_cat in VEHICLE_CATEGORIES:
        dist = levenshtein_distance(cleaned_category_text, known_cat)
        if dist < min_dist:
            min_dist = dist
            closest_category = known_cat
    
    # Accept matches with max 1 character difference
    # This handles small OCR errors like "G1" instead of "C1"
    if min_dist <= 1:
        return closest_category, f"Category '{closest_category}' matched via Levenshtein (Dist={min_dist}) from '{cleaned_category_text}'"
    
    # No match found, return uncertain marker
    return UNCERTAIN_CATEGORY_MARKER, f"Category text '{cleaned_category_text}' (Raw: '{category_text_raw}') not matched"


def _extract_dates_from_row(row, initial_category):
    """
    Extract potential dates from a row of OCR text.
    
    Args:
        row: List of strings representing one row of OCR text
        initial_category: The category assigned to this row
        
    Returns:
        list: List of dictionaries with date information
    """
    today = date.today()
    potential_dates = []
    
    # Check each column in the row for potential date strings
    for col_index, text in enumerate(row):
        # Special case handling:
        # 1. Skip first column if it was already identified as a category
        if col_index == 0 and initial_category != UNCERTAIN_CATEGORY_MARKER:
            continue
        # 2. When category is uncertain, first column might be a date - but verify
        if col_index == 0 and initial_category == UNCERTAIN_CATEGORY_MARKER and not validate_and_format_date(text):
            continue

        # Try to extract a valid date string using helper function
        formatted_date_str = validate_and_format_date(text)
        if formatted_date_str:
            try:
                # Convert to date object for comparison and sorting
                date_obj = datetime.strptime(formatted_date_str, '%d.%m.%Y').date()
                
                # Create date info dictionary
                # 'row_idx' will be set by the caller function
                date_info = {'dt': date_obj, 'str': formatted_date_str, 'row_idx': None}
                potential_dates.append(date_info)
            except ValueError:
                # Skip if datetime conversion fails
                continue
    
    return potential_dates


def _process_individual_rows(structured_rows):
    """
    Process each OCR row for initial category validation and date extraction.
    
    Args:
        structured_rows: List of structured text lines from OCR
        
    Returns:
        tuple: (row_processing_results, all_potential_dates_flat)
    """
    # Initialize result containers
    row_processing_results = []  # Will store processing results for each row
    all_potential_dates_flat = []  # Flat list of all dates from all rows
    
    print(f"  Processing {len(structured_rows)} structured rows for initial validation...")
    
    # Process each row one by one
    for row_index, row in enumerate(structured_rows):
        # Handle empty rows
        if not row:
            row_processing_results.append({
                'row_index': row_index, 
                'original_row': row,
                'initial_category': UNCERTAIN_CATEGORY_MARKER,
                'potential_dates': []
            })
            continue
        
        initial_category = UNCERTAIN_CATEGORY_MARKER
        category_text_raw = ""
        
        try:
            # Step 1: Extract and validate category (typically first column)
            category_text_raw = row[0].strip()
            initial_category, log_message = _validate_category(category_text_raw)
            print(f"      Row {row_index+1}: {log_message}.")
            
            # Step 2: Extract dates from the row
            potential_dates_in_row = _extract_dates_from_row(row, initial_category)
            
            # Step 3: Set row index and add to flat list for later analysis
            for date_info in potential_dates_in_row:
                date_info['row_idx'] = row_index
                all_potential_dates_flat.append(date_info)
            
            print(f"        -> Extracted {len(potential_dates_in_row)} potential dates from row {row_index+1}.")
            
        except IndexError:
            # Handle case where row is empty or has insufficient columns
            print(f"Warning: IndexError processing row {row_index+1}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER
        except Exception as row_err:
            # Catch any other unexpected errors for resilience
            print(f"Warning: Error processing row {row_index+1}: {row_err}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER
        
        # Store row processing result regardless of success or failure
        row_processing_results.append({
            'row_index': row_index,
            'original_row': row,
            'initial_category': initial_category,
            'potential_dates': potential_dates_in_row
        })
    
    return row_processing_results, all_potential_dates_flat


def _infer_missing_categories(row_processing_results):
    """
    Infer missing categories based on known categories in neighboring rows.
    
    This uses a pattern-based inference: if we have rows with categories [X, ???, Z]
    where X and Z are consecutive in the VEHICLE_CATEGORIES list with one category Y
    in between, then Y is likely the missing category.
    
    Args:
        row_processing_results: List of dictionaries with row processing results
        
    Returns:
        int: Number of inferred categories
    """
    print("\n  Attempting to infer missing categories based on row order...")
    num_inferred = 0
    
    # Loop through all rows to find candidates for inference
    for i in range(len(row_processing_results)):
        current_result = row_processing_results[i]
        
        # Check if this row is a candidate for inference:
        # - Category is unknown/uncertain
        # - But the row has dates (partial data)
        if current_result['initial_category'] == UNCERTAIN_CATEGORY_MARKER and current_result['potential_dates']:
            print(f"    Row {i+1}: Candidate for missing category inference (Category='???', Dates found).")
            try:
                # We need both previous and next rows to exist
                if i > 0 and i < len(row_processing_results) - 1:
                    prev_result = row_processing_results[i-1]
                    next_result = row_processing_results[i+1]
                    
                    prev_cat = prev_result['initial_category']
                    next_cat = next_result['initial_category']
                    
                    # Both neighboring rows must have valid categories
                    if prev_cat != UNCERTAIN_CATEGORY_MARKER and next_cat != UNCERTAIN_CATEGORY_MARKER:
                        print(f"      -> Neighbors: Prev='{prev_cat}' (Row {i}), Next='{next_cat}' (Row {i+2}).")
                        
                        # Check if neighbors are in the predefined category list
                        try:
                            prev_idx = VEHICLE_CATEGORIES.index(prev_cat)
                            next_idx = VEHICLE_CATEGORIES.index(next_cat)
                            
                            # If the gap is exactly 2 positions, there's exactly one category between them
                            # Example: B (index 1), ??? (our row), D (index 3) - the missing category would be C (index 2)
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
    return num_inferred


def _aggregate_dates_by_category(row_processing_results):
    """
    Aggregate dates by validated categories.
    
    Groups dates by their associated category, ignoring rows with uncertain categories.
    This prepares data for date pairing in subsequent steps.
    
    Args:
        row_processing_results: List of dictionaries with row processing results
        
    Returns:
        tuple: (category_potential_dates, all_potential_dates)
    """
    # Initialize containers
    category_potential_dates = {}  # Key: category, Value: list of date info dicts
    all_potential_dates = []       # Flat list of all valid dates (from valid categories only)
    
    print("\n  Aggregating dates for validated/inferred categories...")
    for result in row_processing_results:
        final_category = result['initial_category']
        
        # Only process rows with a valid category
        if final_category != UNCERTAIN_CATEGORY_MARKER:
            # Initialize category list if needed
            if final_category not in category_potential_dates:
                category_potential_dates[final_category] = []
                
            # Add dates from this row to the category's list and the flat list
            for date_info in result['potential_dates']:
                # Prevent duplicates within a category
                # This could happen if multiple rows were assigned the same category
                if date_info not in category_potential_dates[final_category]:
                    category_potential_dates[final_category].append(date_info)
                    
                # Add to flat list (duplicates OK - handled by frequency count)
                all_potential_dates.append(date_info)
                
            print(f"    Added {len(result['potential_dates'])} dates for category '{final_category}' (from row {result['row_index']+1})")
    
    return category_potential_dates, all_potential_dates


def _analyze_date_frequencies(all_potential_dates):
    """
    Analyze frequency of dates in the OCR data.
    
    Counts how many times each unique date string appears across all categories.
    High-frequency dates are more likely to be correct and can be prioritized
    when inferring missing dates.
    
    Args:
        all_potential_dates: List of all potential dates found
        
    Returns:
        dict: Dictionary of date strings and their frequencies
    """
    # Count occurrences of each date string
    date_str_counts = {}
    for d_info in all_potential_dates:
        date_str_counts[d_info['str']] = date_str_counts.get(d_info['str'], 0) + 1
    
    print("\n  Analyzing date frequencies (post-inference)...")
    
    # Handle edge cases
    if not all_potential_dates:
        print("    No potential dates found after category validation/inference.")
        return {}
    
    # Log frequency information
    if date_str_counts:
        sorted_date_counts = sorted(date_str_counts.items(), key=lambda item: item[1], reverse=True)
        print("    Detected date frequencies:")
        for d_str, count in sorted_date_counts:
            print(f"      - '{d_str}': {count} time(s)")
    else:
        print("    No dates found to analyze frequency for.")
    
    return date_str_counts


def _find_direct_pairs_and_candidates(category_potential_dates, date_str_counts):
    """
    Find direct date pairs and identify single date candidates.
    
    Implements two key rules:
    1. Direct Pairing: Find valid start+end date pairs within a category
    2. Single Candidate: Identify categories with only a single date
    
    Args:
        category_potential_dates: Dictionary of dates by category
        date_str_counts: Dictionary of date frequencies
        
    Returns:
        tuple: (final_validated_data, single_candidates)
    """
    today = date.today()
    final_validated_data = {}   # Stores categories with complete date pairs
    single_candidates = {}      # Stores categories with one date (for later inference)
    processed_categories_pairing = set()  # Tracks which categories have been processed
    
    print("\n  Assigning initial date pairs and identifying single candidates...")
    
    # Process each category separately
    for category, potential_dates in category_potential_dates.items():
        if category in processed_categories_pairing:
            continue
        
        print(f"    Processing Category for Date Pairing: {category}")
        
        # Sort dates chronologically for easier processing
        potential_dates.sort(key=lambda x: x['dt'])
        
        # -------- RULE 1: Direct Pair Matching --------
        # Find all possible start+end date pairs (start must be <= today, end > start)
        best_direct_pair = None
        possible_pairs = []
        for i in range(len(potential_dates)):
            d1 = potential_dates[i]
            # Valid start dates must be in the past or present
            if d1['dt'] > today:
                continue
                
            # Check each other date as a potential end date
            for j in range(len(potential_dates)):
                if i == j:  # Skip comparing a date with itself
                    continue
                d2 = potential_dates[j]
                
                # End date must be after start date
                if d1['dt'] < d2['dt']:
                    # Score the pair based on frequency of both dates
                    # Higher frequency dates are more likely to be correct
                    freq_score = date_str_counts.get(d1['str'], 0) + date_str_counts.get(d2['str'], 0)
                    possible_pairs.append({'start': d1, 'end': d2, 'score': freq_score})
        
        # If valid pairs found, select the one with highest frequency score
        if possible_pairs:
            best_direct_pair = max(possible_pairs, key=lambda p: p['score'])
            print(f"      -> Rule 1 MATCH: Found direct pair {best_direct_pair['start']['str']} -> {best_direct_pair['end']['str']}")
            final_validated_data[category] = {
                'start_date': best_direct_pair['start']['str'],
                'end_date': best_direct_pair['end']['str']
            }
            processed_categories_pairing.add(category)
            continue
        
        # -------- RULE 2: Single Date Candidate Identification --------
        # If no direct pair was found, check if there's just one date
        
        # Split dates into "start" (<=today) and potential "end" dates (>today)
        category_starts = [d for d in potential_dates if d['dt'] <= today]
        category_potential_ends = [d for d in potential_dates if d not in category_starts]
        
        # Case 1: One start date, no end date
        if len(category_starts) == 1 and len(category_potential_ends) == 0:
            single_candidates[category] = {'known_part': 'start', 'info': category_starts[0]}
            print(f"      -> Rule 2: Found single start candidate: {category_starts[0]['str']}")
            processed_categories_pairing.add(category)
            
        # Case 2: No start date, one end date
        elif len(category_starts) == 0 and len(category_potential_ends) == 1:
            single_candidates[category] = {'known_part': 'end', 'info': category_potential_ends[0]}
            print(f"      -> Rule 2: Found single end candidate: {category_potential_ends[0]['str']}")
            processed_categories_pairing.add(category)
            
        # Case 3: One start and one end date (but Rule 1 didn't match them)
        elif len(category_starts) == 1 and len(category_potential_ends) == 1:
            # Additional check - if they're chronologically valid, pair them
            if category_starts[0]['dt'] < category_potential_ends[0]['dt']:
                print(f"      -> Rule 2 WARNING: Found 1 start ({category_starts[0]['str']}) and 1 potential end ({category_potential_ends[0]['str']}) but Rule 1 failed? Storing pair.")
                final_validated_data[category] = {
                    'start_date': category_starts[0]['str'],
                    'end_date': category_potential_ends[0]['str']
                }
            else:
                # Dates are in wrong order, treat as single start only
                single_candidates[category] = {'known_part': 'start', 'info': category_starts[0]}
                print(f"      -> Rule 2 (Non-chronological): Treating as single start candidate: {category_starts[0]['str']}")
            processed_categories_pairing.add(category)
            
        # Case 4: Ambiguous or no dates
        else:
            print(f"      -> No direct pair and zero or multiple/ambiguous start/end dates for {category}. Cannot infer dates.")
            processed_categories_pairing.add(category)
    
    return final_validated_data, single_candidates


def _infer_date_pairs(single_candidates, all_potential_dates, date_str_counts):
    """
    Infer missing date pairs for categories with only one date.
    
    For each category with only one date (either start or end), this function
    attempts to find the most likely matching date from all dates in the document.
    
    Args:
        single_candidates: Dictionary of categories with single dates
        all_potential_dates: List of all potential dates
        date_str_counts: Dictionary of date frequencies
        
    Returns:
        dict: Dictionary of inferred date pairs by category
    """
    today = date.today()
    inferred_pairs = {}
    
    # Bonus score for dates with matching day and month
    # Example: If start date is 15.06.2018, an end date of 15.06.2028 gets extra points
    DD_MM_BONUS_SCORE = 10
    
    # Filter and sort all valid start dates (once for efficiency)
    all_valid_start_dates_info = sorted([d for d in all_potential_dates if d['dt'] <= today], key=lambda x: x['dt'])
    
    # Log the available dates for debugging
    print(f"    All Valid Start Dates (<= today): {[d['str'] for d in all_valid_start_dates_info]}")
    print(f"    All Potential Dates (for end check): {[d['str'] for d in sorted(all_potential_dates, key=lambda x: x['dt'])]}")
    
    print("\n  Date Inference Pass: Attempting to complete pairs...")
    
    # Process each category with a single date
    for category, candidate_data in single_candidates.items():
        known_part_type = candidate_data['known_part']  # 'start' or 'end'
        known_info = candidate_data['info']             # Date information
        known_dt = known_info['dt']                     # Date object
        known_str = known_info['str']                   # Date string
        
        print(f"    Inferring date partner for {category} (Known {known_part_type}: {known_str})")
        best_partner_info = None
        highest_inference_score = -1
        
        # -------- Case 1: We know the start date, need to find end date --------
        if known_part_type == 'start':
            # Only consider dates later than the known start date
            potential_partners = [p for p in all_potential_dates if p['dt'] > known_dt]
            
            # Score each potential end date
            for partner_info in potential_partners:
                partner_str = partner_info['str']
                partner_dt = partner_info['dt']
                
                # Base score is the frequency of this date in the document
                score = date_str_counts.get(partner_str, 0)
                
                # Bonus for day/month match (common pattern in licenses)
                # Example: 15.06.2018 -> 15.06.2028
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month:
                    score += DD_MM_BONUS_SCORE
                
                # Update best match if score is higher
                if score > highest_inference_score:
                    highest_inference_score = score
                    best_partner_info = partner_info
                # In case of tie, prefer earlier end date (more conservative)
                elif score == highest_inference_score and best_partner_info and partner_dt < best_partner_info['dt']:
                    best_partner_info = partner_info
            
            # Store result based on whether a match was found
            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best End Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                inferred_pairs[category] = {'start_date': known_str, 'end_date': best_partner_info['str']}
            else:
                print(f"      -> DATE INFERENCE FAILED: No suitable end partner found.")
                inferred_pairs[category] = {'start_date': known_str, 'end_date': "--"}
                
        # -------- Case 2: We know the end date, need to find start date --------
        elif known_part_type == 'end':
            # Only consider dates earlier than the known end date and not in future
            potential_partners = [p for p in all_valid_start_dates_info if p['dt'] < known_dt]
            
            # Score each potential start date
            for partner_info in potential_partners:
                partner_str = partner_info['str']
                partner_dt = partner_info['dt']
                
                # Base score is the frequency of this date in the document
                score = date_str_counts.get(partner_str, 0)
                
                # Bonus for day/month match (common pattern in licenses)
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month:
                    score += DD_MM_BONUS_SCORE
                
                # Update best match if score is higher
                if score > highest_inference_score:
                    highest_inference_score = score
                    best_partner_info = partner_info
                # In case of tie, prefer later start date (more conservative)
                elif score == highest_inference_score and best_partner_info and partner_dt > best_partner_info['dt']:
                    best_partner_info = partner_info
            
            # Store result based on whether a match was found
            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best Start Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                inferred_pairs[category] = {'start_date': best_partner_info['str'], 'end_date': known_str}
            else:
                print(f"      -> DATE INFERENCE FAILED: No suitable start partner found.")
                inferred_pairs[category] = {'start_date': "--", 'end_date': known_str}
    
    return inferred_pairs


def _generate_final_output(validated_pairs, inferred_pairs):
    """
    Generate the final dictionary output from validated and inferred pairs.
    
    Combines directly validated pairs with inferred pairs, formats the output,
    and prints a summary table.
    
    Args:
        validated_pairs: Dictionary of validated date pairs by category
        inferred_pairs: Dictionary of inferred date pairs by category
        
    Returns:
        dict: Final output dictionary of date pairs by category
    """
    # Combine validated and inferred pairs
    # If a category appears in both, validated_pairs take precedence
    final_data = {**validated_pairs, **inferred_pairs}
    
    # Format the output for display
    print("\n  --- Final Processed Data (Category & Date Inferred Pairs Included) ---")
    print(f"  {'Category':<8} | {'Start Date':<12} | {'End Date':<12}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12}")
    
    # Initialize final output dictionary
    final_output_dict = {}
    
    # Go through all possible categories in the predefined order
    for category in VEHICLE_CATEGORIES:
        # Get date pair for this category (default to "--" if not found)
        entry = final_data.get(category, {'start_date': "--", 'end_date': "--"})
        start_date = entry['start_date']
        end_date = entry['end_date']
        
        # Only add to final output if at least one date exists
        if start_date != "--" or end_date != "--":
            final_output_dict[category] = {'start_date': start_date, 'end_date': end_date}           
            print(f"  {category:<8} | {start_date:<12} | {end_date:<12}")
    
    return final_output_dict


def process_ocr_rows(structured_rows):
    """
    Applies category and date validation rules to structured OCR rows.
    
    This is the main entry point function that coordinates the entire validation
    and date pairing workflow.
    
    Args:
        structured_rows: List of structured text lines from OCR
        
    Returns:
        Dictionary with validated category and date pairs
    """
    # Step 1: Process each row individually
    # Extract categories and dates from raw OCR text
    row_processing_results, all_potential_dates_flat = _process_individual_rows(structured_rows)
    
    # Step 2: Infer missing categories based on order
    # Use neighbor categories to fill in gaps when possible
    _infer_missing_categories(row_processing_results)
    
    # Step 3: Aggregate dates by category
    # Group dates by their validated/inferred categories
    category_potential_dates, all_potential_dates = _aggregate_dates_by_category(row_processing_results)
    
    # Step 4: Analyze date frequencies
    # Count occurrences of each date for inference rules
    date_str_counts = _analyze_date_frequencies(all_potential_dates)
    if not date_str_counts:
        return {}  # No dates found, return empty result
    
    # Step 5: Find direct pairs and single date candidates
    # Match obvious start/end date pairs, identify lone dates
    direct_pairs, single_candidates = _find_direct_pairs_and_candidates(category_potential_dates, date_str_counts)
    
    # Step 6: Infer date pairs for categories with single dates
    # Complete pairs where only one date exists
    inferred_pairs = _infer_date_pairs(single_candidates, all_potential_dates, date_str_counts)
    
    # Step 7: Generate final output
    # Combine direct and inferred pairs into final result
    return _generate_final_output(direct_pairs, inferred_pairs)
