"""
Configuration module for license OCR pipeline

Loads settings from config.yaml file.
"""

import os
import yaml
from pathlib import Path

# --- Root directory of the project ---
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_FILE_PATH = PROJECT_ROOT / 'config.yaml'
UNCERTAIN_CATEGORY_MARKER = '???'  # Marker for uncertain category

# --- Load configuration from YAML file ---
def load_config():
    """Load configuration from YAML file"""
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file not found at {CONFIG_FILE_PATH}.")
        print("Using default configuration values.")
        return create_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        print("Using default configuration values.")
        return create_default_config()
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
        print("Using default configuration values.")
        return create_default_config()

def create_default_config():
    """Create default configuration if config file is missing or invalid"""
    return {
        'input': {
            'image_path': None,
            'folder_path': None
        },
        'output': {
            'result_file': 'results.json'
        },
        'model': {
            'yolo_model_path': str(PROJECT_ROOT / 'best.pt'),
            'license_class_id': 0,
            'table_class_id': 1
        },
        'ocr': {
            'use_angle_cls': True,
            'language': 'en',
            'use_gpu': False
        },
        'debug': {
            'save_intermediates': True,
            'output_dir': 'debug_output'
        },
        'vehicle_categories': ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J']
    }

# --- Load configuration and set global variables ---
CONFIG = load_config()

# --- Constants for YOLO detection ---
YOLO_MODEL_PATH = Path(CONFIG['model']['yolo_model_path'])
if not YOLO_MODEL_PATH.is_absolute():
    YOLO_MODEL_PATH = PROJECT_ROOT / YOLO_MODEL_PATH
    
LICENSE_CLASS_ID = CONFIG['model']['license_class_id']
TABLE_CLASS_ID = CONFIG['model']['table_class_id']

# --- Constants for category validation ---
VEHICLE_CATEGORIES = CONFIG['vehicle_categories']

# --- Constants for preprocessing and debugging ---
DEBUG_SAVE_INTERMEDIATES = CONFIG['debug']['save_intermediates']
DEBUG_OUTPUT_DIR = Path(CONFIG['debug']['output_dir'])
if not DEBUG_OUTPUT_DIR.is_absolute():
    DEBUG_OUTPUT_DIR = PROJECT_ROOT / DEBUG_OUTPUT_DIR

# --- Input and output paths ---
INPUT_IMAGE_PATH = CONFIG['input'].get('image_path')
INPUT_FOLDER_PATH = CONFIG['input'].get('folder_path')
OUTPUT_RESULT_FILE = CONFIG['output']['result_file']
if not Path(OUTPUT_RESULT_FILE).is_absolute():
    OUTPUT_RESULT_FILE = PROJECT_ROOT / OUTPUT_RESULT_FILE

# --- OCR settings ---
OCR_USE_ANGLE_CLS = CONFIG['ocr']['use_angle_cls']
OCR_LANGUAGE = CONFIG['ocr']['language']
OCR_USE_GPU = CONFIG['ocr']['use_gpu']
