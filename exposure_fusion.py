import cv2
import numpy as np

def load_image(file_path):
    # Load an image from file path and convert it to RGB
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image from {file_path}")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, width, height):
    # Resize image to the given dimensions
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def align_images(images, width, height):
    # Resize images before alignment to ensure they have the same dimensions
    resized_images = [resize_image(img, width, height) for img in images if img is not None]
    # Use MTB algorithm to align images
    aligner = cv2.createAlignMTB()
    aligner.process(resized_images, resized_images)
    return resized_images

def enhance_local_contrast(image):
    # Using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image)
    enhanced_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced_channels)

def exposure_fusion(image1_path, image2_path):
    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Check if images are loaded
    if img1 is None or img2 is None:
        return None

    # Standard size for alignment
    height, width, _ = img1.shape

    # Align images
    images = align_images([img1, img2], width, height)
    if not images or len(images) < 2:
        print("Error aligning images or insufficient images for processing")
        return None

    # Convert images to float32 type for merging
    img1, img2 = [img.astype(np.float32) / 255.0 for img in images]

    # Create an exposure fusion object
    merge_mertens = cv2.createMergeMertens()

    # Merge images
    fused_image = merge_mertens.process([img1, img2])

    # Scale fused image to full 8-bit range
    fused_image = cv2.normalize(fused_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    fused_image = np.clip(fused_image, 0, 255).astype('uint8')

    # Enhance local contrast
    fused_image = enhance_local_contrast(fused_image)

    return fused_image

# Example usage:
fused_img = exposure_fusion('low_exposure.jpg', 'high_exposure.jpg')
if fused_img is not None:
    # Save the fused image
    cv2.imwrite('fused_output.jpg', fused_img)
    print("Fusion complete, image saved as 'fused_output.jpg'")
else:
    print("Fusion failed or no images to process.")






