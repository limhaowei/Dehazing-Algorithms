# Dehazing-Algorithms

A Streamlit-based web application for removing atmospheric haze from images using the AOD-Net (All-in-One Dehazing Network) deep learning model.

## Overview

This application provides a user-friendly web interface for image dehazing, allowing users to upload hazy images and receive dehazed results. The application uses a pre-trained AOD-Net model to process images and remove atmospheric haze while preserving fine details.

## Features

- **Streamlit Interface**: Modern, intuitive web application built with Streamlit
- **AOD-Net Model**: State-of-the-art deep learning model for image dehazing
- **File Validation**: Comprehensive validation including file type, size, and dimension constraints
- **Real-time Processing**: Live feedback during image processing
- **Side-by-side Comparison**: View original and dehazed images side by side
- **Error Handling**: Robust error handling with user-friendly messages
- **Easy Download**: One-click download of dehazed results

## Dataset Information

The model was trained on the NYU2 dataset (Silberman et al., 2012), which includes:

- **Ground Truth Images**: Approximately 1,500 images in the `ori_images/` folder
  - Naming format: `NYU2_x.jpg` where `x` is an integer
- **Hazy Images**: Approximately 27,000 synthetically hazed images in the `hazy_images/` folder
  - Naming format: `NYU2_x_y_z` where `y` and `z` vary to indicate different haze levels
  - Multiple hazed versions are provided for each ground truth image

This dataset provides a comprehensive range of hazy conditions, supporting effective model training for diverse scenarios.

**Reference**: Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). Indoor segmentation and support inference from RGBD images. In A. Fitzgibbon, S. Lazebnik, P. Perona, Y. Sato, & C. Schmid (Eds.), Computer vision – ECCV 2012 (Vol. 7576, pp. 346-360). Springer. https://doi.org/10.1007/978-3-642-33715-4_54

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Dehazing-Algorithms
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

   **Note**: On macOS, you may need to install Apache Arrow for Streamlit:
   ```bash
   brew install apache-arrow
   ```

4. Ensure you have the trained model weights in `saved_model/dehazer_epoch_9.pth`

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**: Click "Choose an image file" and select a hazy image (PNG, JPG, or JPEG)
2. **Wait for Processing**: The model will process your image (this may take a few seconds)
3. **View Results**: Compare the original and dehazed images side by side
4. **Download**: Click the download button to save your dehazed image

### Training the Model

To train the dehazing model (AOD-Net), follow these steps:

1. **Download the Google Colab Notebook** (`AOD_Net.ipynb`)

2. **Set Up Your Environment**: Install the necessary libraries:
```python
!pip install torch torchvision scikit-image opencv-python
```

3. **Upload and Organize Your Dataset**: 
   - Organize your dataset with two folders:
     - `ori_images/`: ~1500 ground truth images (e.g., "NYU2_x.jpg")
     - `hazy_images/`: ~27K hazy images (e.g., "NYU2_x_y_z")
   - Upload the images to Google Colab or link them from cloud storage

4. **Set the Model Configuration**: 
   - In the Args class, configure paths to your dataset, learning rate, weight decay, and other training parameters

5. **Data Preparation**: 
   - The DehazeDataManager class handles loading and splitting the dataset into training, validation, and test sets

6. **Model Definition**: 
   - The AODNet class is initialized and transferred to the GPU for training

7. **Start the Training Process**: 
   - The BaseTrainer class's `fit` method trains the model over multiple epochs
   - Checkpoints are saved after each epoch

8. **Evaluate on the Test Set**: 
   - After training, the model is evaluated on the test set for each epoch
   - The evaluate method computes SSIM and FSIM metrics
   - The best performing model is saved

9. **Save the Trained Model**: 
   - Model checkpoints are saved in the `snapshots` folder
   - Load the best model for further evaluation or inference

## Configuration

Key configuration settings can be found in `config.py`:

- **File Upload Limits**: Maximum file size (10 MB), image dimensions
- **Model Path**: Location of the trained model weights
- **Image Constraints**: Minimum and maximum image dimensions

## Project Structure

```
Dehazing-Algorithms/
├── streamlit_app.py       # Streamlit application (main entry point)
├── config.py               # Configuration constants
├── model.py                # AOD-Net model implementation
├── requirements.txt        # Python dependencies
├── saved_model/            # Trained model weights
│   └── dehazer_epoch_9.pth # Pre-trained model file
└── AOD_Net.ipynb           # Training notebook (optional)
```

## Application Features

- **File Validation**: Checks file type, size, and dimensions
- **Error Handling**: Clear error messages for common issues
- **Progress Indicators**: Visual feedback during processing
- **Image Comparison**: Side-by-side view of original and processed images
- **Format Preservation**: Maintains original image format when possible
