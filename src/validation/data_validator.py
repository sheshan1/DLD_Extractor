"""
Data validation module for OCR results
"""

import re
from datetime import datetime, date
from Levenshtein import distance as levenshtein_distance

from src.utils.helpers import validate_and_format_date
from src.utils.config import VEHICLE_CATEGORIES, UNCERTAIN_CATEGORY_MARKER

def process_ocr_rows(structured_rows):
    """
    Applies category and date validation rules to structured OCR rows.
    
    Args:
        structured_rows: List of structured text lines from OCR
        
    Returns:
        Dictionary with validated category and date pairs
    """
    # Store results per row initially, including failures
    row_processing_results = []
    all_potential_dates_flat = []  # Keep flat list for frequency analysis
    today = date.today()

    print(f"  Processing {len(structured_rows)} structured rows for initial validation...")

    # --- Step 1: Initial Category Validation & Date Extraction per Row ---
    for row_index, row in enumerate(structured_rows):
        if not row:
            row_processing_results.append({
                'row_index': row_index, 
                'original_row': row,
                'initial_category': UNCERTAIN_CATEGORY_MARKER,
                'potential_dates': []
            })
            continue

        initial_category = UNCERTAIN_CATEGORY_MARKER
        potential_dates_in_row = []
        category_text_raw = ""  # Store the raw text OCR'd as category

        try:
            category_text_raw = row[0].strip()
            cleaned_category_text = re.sub(r'[.,;:!?-]$', '', category_text_raw).upper().replace(' ', '')
            # --- Basic OCR corrections for categories ---
            cleaned_category_text = cleaned_category_text.replace('8', 'B').replace('1', 'I')  # Common swaps
            # -------------------------------------------

            if not cleaned_category_text:  # Handle cases where first element is empty after cleaning
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
                if col_index == 0 and initial_category != UNCERTAIN_CATEGORY_MARKER:
                    continue
                # If category was '???', check if col 0 IS a date
                if col_index == 0 and initial_category == UNCERTAIN_CATEGORY_MARKER and not validate_and_format_date(text):
                    continue

                formatted_date_str = validate_and_format_date(text)
                if formatted_date_str:
                    try:
                        date_obj = datetime.strptime(formatted_date_str, '%d.%m.%Y').date()
                        if 1950 <= date_obj.year <= today.year + 30:
                            date_info = {'dt': date_obj, 'str': formatted_date_str, 'row_idx': row_index}
                            potential_dates_in_row.append(date_info)
                            all_potential_dates_flat.append(date_info)  # Add to flat list
                    except ValueError:
                        continue

            print(f"        -> Extracted {len(potential_dates_in_row)} potential dates from row {row_index+1}.")

        except IndexError:
            print(f"Warning: IndexError processing row {row_index+1}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER  # Mark as uncertain
        except Exception as row_err:
            print(f"Warning: Error processing row {row_index+1}: {row_err}. Row: {row}. Skipping category/date extraction.")
            initial_category = UNCERTAIN_CATEGORY_MARKER  # Mark as uncertain

        row_processing_results.append({
            'row_index': row_index,
            'original_row': row,
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
    category_potential_dates = {}  # Rebuild this based on potentially updated results
    all_potential_dates = []       # Rebuild this flat list too

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
                all_potential_dates.append(date_info)  # Use the validated flat list
            print(f"    Added {len(result['potential_dates'])} dates for category '{final_category}' (from row {result['row_index']+1})")

    # --- Step 4: Analyze Frequencies ---
    # Use the 'all_potential_dates' list built in Step 3
    date_str_counts = {}
    for d_info in all_potential_dates:
        date_str_counts[d_info['str']] = date_str_counts.get(d_info['str'], 0) + 1

    print("\n  Analyzing date frequencies (post-inference)...")
    if not all_potential_dates:
        print("    No potential dates found after category validation/inference.")
        return {}
    if date_str_counts:
        sorted_date_counts = sorted(date_str_counts.items(), key=lambda item: item[1], reverse=True)
        print("    Detected date frequencies:")
        for d_str, count in sorted_date_counts:
            print(f"      - '{d_str}': {count} time(s)")
    else:
        print("    No dates found to analyze frequency for.")
        return {}  # Should not happen if all_potential_dates is not empty

    # Prepare global lists needed for date pairing inference
    all_valid_start_dates_info = sorted([d for d in all_potential_dates if d['dt'] <= today], key=lambda x: x['dt'])
    print(f"    All Valid Start Dates (<= today): {[d['str'] for d in all_valid_start_dates_info]}")
    print(f"    All Potential Dates (for end check): {[d['str'] for d in sorted(all_potential_dates, key=lambda x: x['dt'])]}")

    # --- Step 5: Date Pairing (Rule 1) & Single Candidate Identification (Rule 2) ---
    # This now uses the 'category_potential_dates' populated in Step 3
    final_validated_data = {}  # Store confirmed pairs here
    single_candidates = {}  # Store {category: {'known_part': 'start'/'end', 'info': date_info_dict}}
    processed_categories_pairing = set()  # Use a new set for this stage

    print("\n  Assigning initial date pairs and identifying single candidates...")
    # Use the aggregated category_potential_dates dictionary
    for category, potential_dates in category_potential_dates.items():
        # Category here is guaranteed to be valid (not '???')
        if category in processed_categories_pairing:
            continue

        print(f"    Processing Category for Date Pairing: {category}")
        potential_dates.sort(key=lambda x: x['dt'])

        # --- Rule 1: Find direct pair ---
        best_direct_pair = None
        possible_pairs = []
        for i in range(len(potential_dates)):
            d1 = potential_dates[i]
            if d1['dt'] > today:
                continue
            for j in range(len(potential_dates)):
                if i == j:
                    continue
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
    DD_MM_BONUS_SCORE = 10  # Weight for matching day/month

    for category, candidate_data in single_candidates.items():
        if category in final_validated_data:
            continue  # Skip if already assigned in Rule 2 warning case

        known_part_type = candidate_data['known_part']
        known_info = candidate_data['info']
        known_dt = known_info['dt']
        known_str = known_info['str']

        print(f"    Inferring date partner for {category} (Known {known_part_type}: {known_str})")
        best_partner_info = None
        highest_inference_score = -1

        if known_part_type == 'start':
            potential_partners = [p for p in all_potential_dates if p['dt'] > known_dt]
            for partner_info in potential_partners:
                partner_str = partner_info['str']
                partner_dt = partner_info['dt']
                score = date_str_counts.get(partner_str, 0)
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month:
                    score += DD_MM_BONUS_SCORE
                if score > highest_inference_score:
                    highest_inference_score = score
                    best_partner_info = partner_info
                elif score == highest_inference_score and best_partner_info and partner_dt < best_partner_info['dt']:
                    best_partner_info = partner_info

            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best End Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                final_validated_data[category] = {'start_date': known_str, 'end_date': best_partner_info['str']}
            else:
                print(f"      -> DATE INFERENCE FAILED: No suitable end partner found.")
                final_validated_data.setdefault(category, {'start_date': "--", 'end_date': "--"})

        elif known_part_type == 'end':
            potential_partners = [p for p in all_valid_start_dates_info if p['dt'] < known_dt]
            for partner_info in potential_partners:
                partner_str = partner_info['str']
                partner_dt = partner_info['dt']
                score = date_str_counts.get(partner_str, 0)
                if partner_dt.day == known_dt.day and partner_dt.month == known_dt.month:
                    score += DD_MM_BONUS_SCORE
                if score > highest_inference_score:
                    highest_inference_score = score
                    best_partner_info = partner_info
                elif score == highest_inference_score and best_partner_info and partner_dt > best_partner_info['dt']:
                    best_partner_info = partner_info

            if best_partner_info:
                print(f"      -> DATE INFERRED MATCH: Best Start Partner '{best_partner_info['str']}' (Score: {highest_inference_score})")
                final_validated_data[category] = {'start_date': best_partner_info['str'], 'end_date': known_str}
            else:
                print(f"      -> DATE INFERENCE FAILED: No suitable start partner found.")
                final_validated_data.setdefault(category, {'start_date': "--", 'end_date': "--"})

    # --- Final Output Generation ---
    print("\n  --- Final Processed Data (Category & Date Inferred Pairs Included) ---")
    print(f"  {'Category':<8} | {'Start Date':<12} | {'End Date':<12}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12}")
    final_output_dict = {}
    
    for category in VEHICLE_CATEGORIES:
        entry = final_validated_data.get(category, {'start_date': "--", 'end_date': "--"})
        start_date = entry['start_date']
        end_date = entry['end_date']
        
        # Only add to final output if at least one date exists
        if start_date != "--" or end_date != "--":
            final_output_dict[category] = {'start_date': start_date, 'end_date': end_date}           
            print(f"  {category:<8} | {start_date:<12} | {end_date:<12}")

    return final_output_dict
