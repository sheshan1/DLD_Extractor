# License OCR Pipeline

A modular Python system for processing driver's license documents through computer vision and OCR. The pipeline detects license components, extracts table data, and produces structured output of vehicle categories with their validity dates.
## Features

- **License Document Detection**: Uses YOLO v8 object detection to identify license and table components
- **Image Preprocessing**: Handles rotation correction, skew detection, and image enhancement for OCR
- **Table Structure Analysis**: Detects table rows and columns to improve OCR layout understanding
- **OCR Processing**: Extracts text from license tables using PaddleOCR with high accuracy
- **Data Validation**: Processes and validates extracted vehicle categories and dates with fuzzy matching
- **Structured Output**: Returns JSON with vehicle category permissions and validity dates
- **Customizable Processing**: Easily extend or modify the pipeline via configuration
- **Batch Processing**: Process single images or entire directories of license documents
- **Debug Visualization**: Generate visual debug outputs to diagnose processing issues

## Project Structure

```
DLD_Extractor/
├── best.pt                 # YOLO model file for component detection
├── main.py                 # Main entry point script
├── config.yaml             # Configuration file for all pipeline settings
├── requirements.txt        # Dependencies list
├── yolo-training.ipynb     # Notebook for training YOLO model
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
└── debug_output/           # Debug output directory with processed image components
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

4. Download the YOLO model:
   - The repository already includes the trained YOLO model (`best.pt`)
   - If you need to train a new model, you can use the included `yolo-training.ipynb` notebook

## Usage

The pipeline uses a configuration file (`config.yaml`). Edit this file to specify your inputs and outputs, then run the pipeline with a single command.

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
  license_class_id: 0                  # Class ID for license component
  table_class_id: 1                    # Class ID for table component

# OCR configuration
ocr:
  use_angle_cls: true                  # Whether to use angle classification
  language: "en"                       # OCR language
  use_gpu: false                       # Whether to use GPU for OCR

# Debug configuration
debug:
  save_intermediates: true             # Whether to save intermediate images
  output_dir: "debug_output"           # Directory for debug output
  
# Vehicle categories to detect
vehicle_categories:
  - "A1"
  - "A"
  - "B1"
  - "B"
  # ... other categories
```

### Running the Pipeline

After configuring the `config.yaml` file, simply run:

```bash
python3 main.py
```

### Processing Multiple Images

To process all images in the `test_images` directory:

1. Edit `config.yaml`:
   ```yaml
   input:
     folder_path: "test_images"
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

### Viewing Results

The results are saved to the JSON file specified in the config (default: `results.json`). You can view them in the JSON file.

### Quick Start

To quickly get started with DLD Extractor:

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/sheshan1/DLD_Extractor.git
   cd DLD_Extractor
   pip install -r requirements.txt
   ```

2. Edit the configuration file to point to your image:
   ```yaml
   input:
     image_path: "path/to/your/license_image.jpg"
   ```

3. Run the extraction pipeline:
   ```bash
   python main.py
   ```

4. Check the results in `results.json`

## Example Configurations

#### Processing a Single Image

```yaml
input:
  image_path: "test_images/181195.jpg"
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
   - Applies multiple preprocessing techniques to improve OCR accuracy

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
    "B": {
      "start_date": "01.01.2020",
      "end_date": "01.01.2025"
    },
    "C1": {
      "start_date": "01.01.2020",
      "end_date": "01.01.2025"
    }
  },
  "another_image.jpg": {
    "A": {
      "start_date": "15.03.2021",
      "end_date": "15.03.2031"
    },
    "B": {
      "start_date": "15.03.2021",
      "end_date": "15.03.2031"
    }
  }
}
```

For each image processed, the JSON output contains:
- The image filename as the primary key
- Each detected vehicle category (A1, A, B1, B, etc.) as sub-keys
- For each category, the corresponding start and end dates
- Dates are formatted in DD.MM.YYYY format for consistency
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

### Detection Module (`src/detection/yolo_detector.py`)

The detection module uses YOLO to identify license components:

- `preprocess_for_YOLO()`: Prepares images for YOLO detection by resizing and normalizing
- `extract_license_components()`: Runs YOLO detection and extracts license & table components
- Handles different image formats, orientations, and sizes
- Supports debug output of intermediate detection results

### Preprocessing Module (`src/preprocessing/image_processor.py`)

The preprocessing module handles image enhancements:

- `preprocess_image()`: Applies configurable image processing techniques:
  - Grayscale conversion
  - Adaptive thresholding
  - Noise reduction
  - Contrast enhancement (CLAHE)
  - Histogram equalization
- `preprocess_for_ocr()`: Detects orientation and rotates images as needed
- `detect_table_structure()`: Identifies table rows and columns for improved OCR
- `deskew_image()`: Corrects image skew for better OCR results

### OCR Module (`src/ocr/ocr_engine.py`)

The OCR module handles text extraction:

- `initialize_ocr_engine()`: Sets up PaddleOCR with configurable parameters
- `group_lines()`: Groups OCR results into structured text lines based on spatial layout
- `sort_text_blocks()`: Arranges detected text in reading order
- `run_ocr_pipeline()`: Coordinates preprocessing, OCR execution, and post-processing

### Validation Module (`src/validation/data_validator.py`)

The validation module processes extracted text:

- `process_ocr_rows()`: Main validation function analyzing extracted rows
- `_validate_category()`: Validates and normalizes vehicle categories using fuzzy matching
- `_extract_dates_from_row()`: Extracts and validates date formats
- `_pair_dates()`: Pairs start and end dates for each vehicle category
- `_infer_missing_data()`: Uses context to fill in missing information
- Extensive error handling for various OCR error cases

### Utilities

- `src/utils/config.py`: Central configuration parameter management
- `src/utils/helpers.py`: Common utility functions for date validation, text processing, and image handling

## Debug Output

When `DEBUG_SAVE_INTERMEDIATES` is enabled, the pipeline saves intermediate images to the `debug_output` directory, including:

- License component crops
- Table component crops
- Rotated table images
- Preprocessed OCR images

## Requirements

- Python 3.8+
- Ultralytics YOLO v8+
- PaddleOCR 2.6+ & PaddlePaddle 2.4+
- OpenCV 4.5+
- Pytesseract 0.3.8+
- Python-Levenshtein 0.20+
- Pillow 8.0+
- NumPy 1.19+
- Matplotlib 3.5+
- PyYAML 6.0+

All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

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
   - Validate that the YAML indentation is correct

2. **YOLO Detection Issues**:
   - Check that the model file specified in `model.yolo_model_path` exists
   - Verify input images are readable and not corrupted
   - If detection is poor, consider retraining the model using the `yolo-training.ipynb`
   - Ensure there is sufficient lighting and contrast in the input images
   - Try different confidence thresholds if detections are missed

3. **OCR Accuracy Issues**:
   - Try different preprocessing methods in the configuration:
     ```yaml
     ocr:
       preprocessing_methods:
         - "gray"
         - "denoise"
         - "adaptive_threshold"
     ```
   - For many images, using only "gray" preprocessing provides better results
   - Adjust confidence thresholds in `ocr_engine.py`
   - Consider using GPU acceleration for better OCR performance:
     ```yaml
     ocr:
       use_gpu: true
     ```

4. **Rotation/Skew Detection Issues**:
   - Check that Tesseract is properly installed
   - Ensure the license component is being detected
   - For manual override, you can modify the preprocessing module
   - Check the debug output images to see if rotation is correctly detected
   - If rotation is incorrectly detected, force a specific orientation in the code

5. **Performance Issues**:
   - For faster processing on capable hardware, enable GPU:
     ```yaml
     ocr:
       use_gpu: true
     ```
   - Reduce resolution of input images if performance is still an issue
   - Process images in smaller batches
   - Close other applications when processing large numbers of images

## Debug Output Analysis

When debugging is enabled, the following files are generated in the debug directory for each image:

- `license_crop.jpg`: The detected license component
- `table_crop.jpg`: The detected table component
- `deskewed_image.jpg`: The rotation-corrected table image
- `ocr_preprocessed.jpg`: The table after preprocessing for OCR
- `horizontal_lines.jpg`: Detected horizontal table lines
- `vertical_lines.jpg`: Detected vertical table lines
- `table_structure.jpg`: Visualization of the detected table structure

These files are useful for diagnosing issues in the detection, preprocessing, and OCR stages.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for text recognition
- [OpenCV](https://opencv.org/) contributors for image processing capabilities
- [Python-Levenshtein](https://github.com/ztane/python-Levenshtein) for fuzzy text matching

## Development and Contributing

### Project Setup for Development

1. Fork and clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8
   ```

3. Set up pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### YOLO Model Training

The project includes a notebook (`yolo-training.ipynb`) for training the YOLO model:

1. Prepare labeled training data (license and table components)
2. Configure the notebook parameters
3. Run the training process
4. Export the trained model as `best.pt`

### Testing

Run tests with:
```bash
pytest tests/
```

### Future Enhancements

- Support for additional document types
- Improved handling of damaged or low-quality images
- Improve the orientation detection and better handling of skewed images
- Machine learning-based validation for extracted data
- Web interface for interactive processing
- Support for additional languages

## License

This project is licensed under the MIT License - see the LICENSE file for details.
