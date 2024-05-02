import os
import cv2
from image_loader import load_image
from image_processor import ImageProcessor

def perform_exposure_fusion(input_folder, output_folder, image1_name, image2_name):
    processor = ImageProcessor()

    img1 = load_image(os.path.join(input_folder, image1_name))
    img2 = load_image(os.path.join(input_folder, image2_name))

    if img1 is None or img2 is None:
        print("One or more images could not be loaded.")
        return

    height, width, _ = img1.shape
    aligned_images = processor.align_images([img1, img2], width, height)
    fused_image = processor.merge_images(aligned_images)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_image_path = os.path.join(output_folder, f"fused_{image1_name}")
    cv2.imwrite(output_image_path, cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR))
    print(f"Fusion complete, image saved as '{output_image_path}'")

if __name__ == "__main__":
    input_folder = 'input_images'
    output_folder = 'result_images'
    image1_name = 'low_exposure_1.jpg'
    image2_name = 'high_exposure_1.jpg'
    perform_exposure_fusion(input_folder, output_folder, image1_name, image2_name)
    

    image3_name = 'low_exposure.jpg'
    image4_name = 'high_exposure.jpg'
    perform_exposure_fusion(input_folder, output_folder, image3_name, image4_name)
