"""
Utility functions for license OCR pipeline
"""

import re
from datetime import datetime, date, timedelta
from Levenshtein import distance as levenshtein_distance

def validate_and_format_date(text):
    """
    Validates text against various date patterns, returns 'DD.MM.YYYY' or None.
    
    Args:
        text: String to validate as a date
        
    Returns:
        Formatted date string in DD.MM.YYYY format or None if not a valid date
    """
    if not isinstance(text, str):
        return None
        
    cleaned_text = text.strip().replace(' ', '')
    cleaned_text = cleaned_text.replace('O', '0').replace('o', '0')
    cleaned_text = cleaned_text.replace('l', '1').replace('I', '1').replace('!', '1')
    cleaned_text = cleaned_text.replace('S', '5').replace('s', '5')
    cleaned_text = cleaned_text.replace('B', '8')
    cleaned_text = cleaned_text.replace('g', '9').replace('q', '9')
    
    patterns = [
        {'regex': r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', 'groups': (1, 2, 3)}, 
        {'regex': r'^(\d{1,2})\.(\d{1,2})(\d{4})$', 'groups': (1, 2, 3)},
        {'regex': r'^(\d{1,2})(\d{1,2})\.(\d{4})$', 'groups': (1, 2, 3)}, 
        {'regex': r'^(\d{4})\.(\d{1,2})\.(\d{1,2})$', 'groups': (3, 2, 1)},
        {'regex': r'^(\d{2})(\d{2})(\d{4})$', 'groups': (1, 2, 3)}, 
        {'regex': r'^(\d{4})(\d{2})(\d{2})$', 'groups': (3, 2, 1)},
        {'regex': r'^(\d{2})(\d{2})(\d{2})$', 'groups': (1, 2, 3), 'infer_century': True},
    ]
    
    for p in patterns:
        match = re.fullmatch(p['regex'], cleaned_text)
        if match:
            try:
                day_idx, month_idx, year_idx = p['groups']
                dd_str = match.group(day_idx)
                mm_str = match.group(month_idx)
                yyyy_str = match.group(year_idx)
                
                if p.get('infer_century', False):
                    year = int(yyyy_str)
                    current_year_short = date.today().year % 100
                    yyyy_str = f"20{yyyy_str.zfill(2)}" if year <= current_year_short + 10 else f"19{yyyy_str.zfill(2)}"
                
                dd_str = dd_str.zfill(2)
                mm_str = mm_str.zfill(2)
                
                if len(yyyy_str) != 4:
                    continue
                    
                dt_obj = datetime(int(yyyy_str), int(mm_str), int(dd_str))
                return dt_obj.strftime('%d.%m.%Y')
            except (ValueError, IndexError):
                continue
    
    return None

def add_years_to_date(source_date, years_to_add):
    """
    Safely adds years to a date object, handling leap years.
    
    Args:
        source_date: Date object to add years to
        years_to_add: Number of years to add
        
    Returns:
        New date object with years added, or None if operation fails
    """
    try:
        new_year = source_date.year + years_to_add
        return source_date.replace(year=new_year)
    except ValueError:
        if source_date.month == 2 and source_date.day == 29:
            try:
                return date(new_year, 2, 28)
            except ValueError:
                return None
        else:
            return None
