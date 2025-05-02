# License OCR Pipeline

A modular Python system for processing driver's license documents through computer vision and OCR. The pipeline detects license components, extracts table data, and produces structured output of vehicle categories with their validity dates.
## Features

- **License Document Detection**: Uses YOLO object detection to identify license and table components
- **Image Preprocessing**: Handles rotation correction and image enhancement for OCR
- **OCR Processing**: Extracts text from the license table
- **Data Validation**: Processes and validates extracted vehicle categories and dates
- **Structured Output**: Returns JSON with vehicle category permissions and validity dates
- **Customizable Processing**: Easily extend or modify the pipeline via configuration

## Project Structure

```
NewTry/
├── best.pt                 # YOLO model file
├── main.py                 # Main entry point script
├── config.yaml             # Configuration file for all pipeline settings
├── requirements.txt        # Dependencies list
├── src/                    # Source code directory
│   ├── detection/          # YOLO detection module
│   │   └── yolo_detector.py
│   ├── ocr/                # OCR processing module
│   │   └── ocr_engine.py
│   ├── preprocessing/      # Image preprocessing module
│   │   └── image_processor.py
│   ├── utils/              # Utility functions
│   │   ├── config.py       # Configuration loader
│   │   └── helpers.py
│   └── validation/         # Data validation module
│       └── data_validator.py
├── test_images/            # Test images directory
└── debug_output/           # Debug output directory
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sheshan1/DLD_Extractor.git
   cd DLD_Extractor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The pipeline uses a configuration file (`config.yaml`). Simply edit the configuration file to specify your inputs and outputs, then run the pipeline.

### Configuration File

The `config.yaml` file contains all pipeline settings:

```yaml
# Input configuration
input:
  image_path: "test_images/image.jpg"  # Path to a single image file
  # OR
  # folder_path: "test_images"         # Path to a folder with multiple images

# Output configuration
output:
  result_file: "results.json"          # Path to the output JSON file

# YOLO model configuration
model:
  yolo_model_path: "best.pt"           # Path to YOLO model file

# Debug configuration
debug:
  save_intermediates: true             # Whether to save intermediate images
  output_dir: "debug_output"           # Directory for debug output
```

### Running the Pipeline

After configuring the `config.yaml` file, simply run:

```bash
python3 main.py
```

### Example Configurations

#### Processing a Single Image

```yaml
input:
  image_path: "test_images/170349.jpg"
```

#### Processing All Images in a Directory

```yaml
input:
  folder_path: "test_images"
```

#### Changing the Output File

```yaml
output:
  result_file: "custom_results.json"
```

## Pipeline Flow

1. **Component Detection**:
   - The YOLO model detects license document components
   - Extracts license and table regions as separate images

2. **Image Preprocessing**:
   - Determines document orientation using the license component (It has more characters compared to the table)
   - Rotates and preprocesses the table image for OCR

3. **OCR Processing**:
   - Applies PaddleOCR to extract text from the table
   - Groups OCR results into structured text lines

4. **Data Validation**:
   - Identifies vehicle categories (A1, A, B1, B, etc.)
   - Validates and formats dates
   - Pairs start and end dates for each vehicle category
   - Uses inference rules to complete partial data

5. **Output Generation**:
   - Creates structured JSON with category and date information

## Output Format

The pipeline generates a JSON file with the following structure:

```json
{
  "image_filename.jpg": {
    "A1": {
      "start_date": "01.01.2020",
      "end_date": "01.01.2025"
    },
    "A": {
      "start_date": "01.01.2020",
      "end_date": "01.01.2025"
    },
    "B1": {
      "start_date": "01.01.2020",
      "end_date": "01.01.2025"
    },
    ...
  },
  "another_image.jpg": {
    ...
  }
}
```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`) for all settings, making it easy to configure without modifying code.

### Configuration File Structure

```yaml
# License OCR Pipeline Configuration

# Input configuration
input:
  # Use one of the following:
  image_path: "test_images/170349.jpg"  # Path to a single image file
  # folder_path: "test_images"          # Path to a folder with multiple images

# Output configuration
output:
  result_file: "results.json"  # Path to the output JSON file

# YOLO model configuration
model:
  yolo_model_path: "best.pt"   # Path to YOLO model file
  license_class_id: 0          # Class ID for license component
  table_class_id: 1            # Class ID for table component

# OCR configuration
ocr:
  use_angle_cls: true          # Whether to use angle classification
  language: "en"               # OCR language
  use_gpu: false               # Whether to use GPU for OCR
  preprocessing_methods:       # List of preprocessing methods to apply
    - "gray"
    # - "adaptive_threshold"
    # - "denoise"
    # - "clahe"
    # - "hist_eq"

# Debug configuration
debug:
  save_intermediates: true     # Whether to save intermediate images
  output_dir: "debug_output"   # Directory for debug output

# Vehicle categories to look for
vehicle_categories:
  - "A1"
  - "A"
  - "B1"
  # ... other categories
```

### Key Configuration Parameters

- **Input Settings**:
  - `input.image_path`: Path to a single image to process
  - `input.folder_path`: Path to a folder containing multiple images to process

- **Output Settings**:
  - `output.result_file`: Path where the JSON results will be saved

- **YOLO Model Settings**:
  - `model.yolo_model_path`: Path to the YOLO model file
  - `model.license_class_id`: Class ID for license component in the YOLO model
  - `model.table_class_id`: Class ID for table component in the YOLO model

- **OCR Settings**:
  - `ocr.preprocessing_methods`: List of preprocessing techniques to apply to images
  - `ocr.use_angle_cls`: Whether to use PaddleOCR's angle classification
  - `ocr.language`: Language to use for OCR
  - `ocr.use_gpu`: Whether to use GPU for OCR processing

- **Debug Settings**:
  - `debug.save_intermediates`: Whether to save intermediate processing images
  - `debug.output_dir`: Directory for debug output

- **Vehicle Categories**:
  - `vehicle_categories`: List of valid vehicle categories to detect

## Module Descriptions

### Detection Module

The detection module uses YOLO to identify license components:

- `preprocess_for_YOLO()`: Prepares images for YOLO detection
- `extract_license_components()`: Runs YOLO and extracts license & table components

### Preprocessing Module

The preprocessing module handles image enhancements:

- `preprocess_image()`: Applies various image processing techniques
- `preprocess_for_ocr()`: Detects orientation and rotates images as needed

### OCR Module

The OCR module handles text extraction:

- `group_lines()`: Groups OCR results into text lines
- `run_ocr_pipeline()`: Coordinates preprocessing and OCR execution

### Validation Module

The validation module processes extracted text:

- `process_ocr_rows()`: Main validation function analyzing extracted rows
- Multiple validation rules for category recognition
- Date pairing algorithms for associating start/end dates

### Utilities

- `config.py`: Central configuration parameters
- `helpers.py`: Common utility functions for date validation

## Debug Output

When `DEBUG_SAVE_INTERMEDIATES` is enabled, the pipeline saves intermediate images to the `debug_output` directory, including:

- License component crops
- Table component crops
- Rotated table images
- Preprocessed OCR images

## Requirements

- Python 3.8+
- Ultralytics YOLO
- PaddleOCR & PaddlePaddle
- OpenCV
- Pytesseract
- Python-Levenshtein
- Pillow
- NumPy
- Matplotlib

## Extending the Pipeline

### Modifying the Configuration

The easiest way to customize the pipeline is by editing the `config.yaml` file:

```yaml
# Add preprocessing methods
ocr:
  preprocessing_methods:
    - "gray"
    - "adaptive_threshold"
    - "denoise"
    - "clahe"

# Change OCR language
ocr:
  language: "fr"  # For French documents

# Configure for GPU usage
ocr:
  use_gpu: true
```

### Adding New Preprocessing Methods

1. Add your method to `src/preprocessing/image_processor.py` in the `preprocess_image()` function:

```python
if 'my_new_method' in methods and img is not None:
    # Apply your custom preprocessing technique
    img = my_custom_function(img)
```

2. Enable it in the configuration file:

```yaml
ocr:
  preprocessing_methods:
    - "gray"
    - "my_new_method"
```

### Customizing OCR Parameters

1. Adjust OCR settings in the configuration file:

```yaml
ocr:
  use_angle_cls: true
  language: "en"
  use_gpu: false
```

2. For more advanced customization, modify the OCR engine code in `src/ocr/ocr_engine.py`.

### Enhancing Data Validation

Add new validation rules or category matching logic in `src/validation/data_validator.py`.

### Processing Custom Document Types

To adapt the pipeline for different document formats:
1. Retrain the YOLO model with new component labels
2. Update the `model` section in the configuration:

```yaml
model:
  yolo_model_path: "custom_model.pt"
  custom_component_id: 2  # ID for your new component
```

## Troubleshooting

### Common Issues

1. **Configuration Issues**:
   - Ensure the `config.yaml` file exists and has valid YAML syntax
   - Verify file paths in the configuration are correct for your environment
   - Check that either `image_path` or `folder_path` is specified (but not both)

2. **YOLO Detection Issues**:
   - Check that the model file specified in `model.yolo_model_path` exists
   - Verify input images are readable and not corrupted
   - If detection is poor, consider retraining the model (can modify the YOLO_training.ipynb)

3. **OCR Accuracy Issues**:
   - Try different preprocessing methods in the configuration: (using only "gray" had the better results in my case)
     ```yaml
     ocr:
       preprocessing_methods:
         - "gray"
         - "denoise"
         - "adaptive_threshold"
     ```
   - Adjust confidence thresholds in `ocr_engine.py`

4. **Rotation Detection Issues**:
   - Check Tesseract is properly installed
   - Ensure the license component is being detected
   - If rotation is incorrectly detected, force a specific orientation in the code

5. **Performance Issues**:
   - For faster processing on capable hardware, enable GPU:
     ```yaml
     ocr:
       use_gpu: true
     ```
   - Reduce resolution of input images if performance is still an issue

## Acknowledgments

- Ultralytics YOLO for object detection
- PaddleOCR for text recognition
- OpenCV contributors for image processing capabilities
