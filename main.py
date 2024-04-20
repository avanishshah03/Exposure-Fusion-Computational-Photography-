import cv2
import numpy as np
from image_loader import load_image
from image_processor import ImageProcessor

def perform_exposure_fusion(image1_path, image2_path):
    processor = ImageProcessor()
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

    return fused_image

# Example usage:
fused_img = perform_exposure_fusion('low_exposure.jpg', 'high_exposure.jpg')
if fused_img is not None:
    cv2.imwrite('fused_output.jpg', fused_img)
    print("Fusion complete, image saved as 'fused_output.jpg'")
else:
    print("Fusion failed or no images to process.")
