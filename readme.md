# S3A to YOLOv8 Converter

This project is designed to convert the proprietary S3A format used for the FPIC dataset into the YOLOv8 format, enabling the training of object detection and segmentation algorithms using the YOLOv8 file format.

## Project Overview

The project consists of three main components:

1. **Image Patching:** Images and annotations are divided into equally sized patches to facilitate processing and training of neural networks.
2. **Data Conversion:** The patched images and corresponding annotations are converted from the S3A format to the YOLOv8 format.
3. **Data Splitting:** The converted data is split into training, validation, and test datasets.

Additionaly the project provides following three sub components:

1. **HSI + Clahe Conversion:**
Images are converted from RGB to HSI and CLAHE contrast adjustment is applied.
2. **Designator Copying:**
Copy Designators from an existing annotation file to a target annotation file.
3. **Designator Replacement:**
Replace existing Designators from annotation files with newly specified Designators.


## Installation

To install Automated_PCB_based_Circuit_Reconstruction and its dependencies, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/VivDue/s3a_yolo_converter.git
    ```

2. Navigate to the cloned directory and replace your path with your save path:

    ```bash
    cd your_path/Automated_PCB_based_Circuit_Reconstruction
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment:

    - **Windows:**

    ```bash
    .venv\Scripts\activate.bat
    ```

    - **Linux/macOS:**

    ```bash
    source .venv/bin/activate
    ```

5. Install the required dependencies and replace the your_path with your save path:

    - `Using` requirements.txt:
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r your_path/requirements.txt
    ```

    - `Alternatively`, install packages individually:
    If the requirements.txt installation fails, you can install the necessary packages manually:
    ```bash
    python -m pip install opencv-python
    python -m pip install scikit-learn
    python -m pip install tqdm
    python -m pip install ipykernel
    ```

## Example Workflow

A complete workflow pipeline example can be found in `example.ipynb`. This notebook provides step-by-step instructions on how to use the scripts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
