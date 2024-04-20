# Exposure Fusion Project

## Introduction
This project implements an exposure fusion technique using Python, aimed at combining multiple images with different exposures into a single image with a more balanced exposure throughout. It is particularly useful for scenarios where high dynamic range (HDR) imaging is required but HDR capture is not feasible.

## Use Cases
Exposure fusion is beneficial in several photographic scenarios:
- **Architecture Photography**: Enhancing interior shots where bright windows often cause exposure imbalance.
- **Landscape Photography**: Balancing sky and land where lighting conditions vary significantly across the scene.
- **Night Cityscapes**: Combining lights and darks in nighttime city photography to capture detail in both brightly lit areas and shadows.

## Project Structure
The project is structured into several modules to handle different aspects of the image processing pipeline:
- `image_loader.py`: Manages the loading and initial preprocessing of images.
- `image_processor.py`: Contains all methods related to image alignment, merging, and contrast enhancement.
- `main.py`: Orchestrates the reading of images, processing them through the pipeline, and saving the results.

## Installation
Before running the project, ensure that Python and OpenCV are installed on your system. You can install OpenCV using pip:
```bash
pip install opencv-python-headless
```

##Team Members
- Avanish Shah
- Dev Patel
- Rijul Ranjan

