import os
import cv2
import numpy as np
from image_loader import load_image
from image_processor import ImageProcessor

def perform_exposure_fusion(input_folder, output_folder, image1_name, image2_name):
    processor = ImageProcessor()
    
    # Construct paths to the input images
    image1_path = os.path.join(input_folder, image1_name)
    image2_path = os.path.join(input_folder, image2_name)

    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    if img1 is None or img2 is None:
        print("One or more images could not be loaded.")
        return None

    height, width, _ = img1.shape
    aligned_images = processor.align_images([img1, img2], width, height)
    
    if not aligned_images or len(aligned_images) < 2:
        print("Error aligning images or insufficient images for processing.")
        return None
    
    fused_image = processor.merge_images(aligned_images)
    fused_image = np.clip(fused_image, 0, 255).astype('uint8')
    fused_image = processor.enhance_local_contrast(fused_image)

    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct output image path
    output_image_path = os.path.join(output_folder, f"fused_{image1_name}")

    # Save the fused image
    cv2.imwrite(output_image_path, fused_image)
    print(f"Fusion complete, image saved as '{output_image_path}'")
    return output_image_path

# Directory and file configuration
input_folder = 'input_images'
output_folder = 'result_images'
image1_name = 'low_exposure_1.jpg'
image2_name = 'high_exposure_1.jpg'

# Perform fusion
result_path = perform_exposure_fusion(input_folder, output_folder, image1_name, image2_name)
if result_path:
    print("Image processing successful.")
else:
    print("Image processing failed.")

