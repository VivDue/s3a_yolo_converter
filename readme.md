Sure! Here's the README file in English:

---

# S3A to YOLOv8 Converter

This project is designed to convert the proprietary S3A format used for the FPIC dataset into the YOLOv8 format, enabling the training of object detection and segmentation algorithms using the YOLOv8 file format.

## Project Overview

The project consists of three main components:

1. **Image Patching:** Images are divided into equally sized patches to facilitate processing and training of neural networks.
2. **Data Conversion:** The patched images and corresponding annotations are converted from the S3A format to the YOLOv8 format.
3. **Data Splitting:** The converted data is split into training, validation, and test datasets.

## Project Structure

The project includes the following files and directories:

- **`spa_converter.py`**: Contains the logic for converting S3A format to YOLOv8 format.
- **`data_spliter.py`**: Splits the converted data into training, validation, and test datasets.
- **`spa_patch_creator.py`**: Creates patches from the original images and their associated annotations.
- **`example.ipynb`**: A Jupyter notebook that demonstrates how to use the scripts in a typical workflow pipeline.
- **`requirements.txt`**: A list of Python libraries required for the project.
- **`spa_sample_input/`**: A directory containing sample images and annotations in S3A format, which can be used as input for the scripts.

## Installation

To set up the project on your local system, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your_repository_url>
   cd <your_directory>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Image Patching

To split images into patches, use the `spa_patch_creator.py` script:

```bash
python spa_patch_creator.py --ann_dir <path_to_annotations> --img_dir <path_to_images> --output_dir <output_path> --patch_size <patch_size_in_pixels>
```

### 2. Data Conversion

Use the `spa_converter.py` script to convert the patches and annotation data into YOLOv8 format:

```bash
python spa_converter.py --input_dir <path_to_patches> --output_dir <output_path>
```

### 3. Data Splitting

Finally, use the `data_spliter.py` script to divide the converted data into training, validation, and test datasets:

```bash
python data_spliter.py --input_dir <path_to_converted_data> --output_dir <output_path> --split_ratio <train_val_test_split_ratio>
```

### Example Workflow

A complete workflow pipeline example can be found in `example.ipynb`. This notebook provides step-by-step instructions on how to use the scripts.

## Contributing

Contributions are welcome! Please open an issue to report bugs or suggest new features, or submit a pull request directly.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive description of your project and guides users through installation and usage. You can adjust or expand it as needed.