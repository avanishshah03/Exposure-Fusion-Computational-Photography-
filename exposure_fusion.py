import cv2
import numpy as np

def load_image(file_path):
    # Load an image from file path and convert it to RGB
    image = cv2.imread(file_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, width, height):
    # Resize image to the given dimensions
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def exposure_fusion(image1_path, image2_path):
    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Check if images are loaded
    if img1 is None or img2 is None:
        print("Error loading images")
        return None

    # Ensure both images are the same size
    height, width, _ = img1.shape
    img1 = resize_image(img1, width, height)
    img2 = resize_image(img2, width, height)

    # Convert images to float32 type
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Create an exposure fusion object
    merge_mertens = cv2.createMergeMertens()

    # Merge images
    fused_image = merge_mertens.process([img1, img2])

    # Diagnostic: check the range of fused image values
    print(f"Fused image range: min={fused_image.min()}, max={fused_image.max()}")

    # Normalize the fused image
    fused_image -= fused_image.min()  # shift to 0
    fused_image /= fused_image.max()  # scale to 1
    fused_image = (fused_image * 255).astype('uint8')  # scale to [0, 255]

    return fused_image

# Example usage:
fused_img = exposure_fusion('low_exposure.jpg', 'high_exposure.jpg')
if fused_img is not None:
    # Save the fused image
    cv2.imwrite('fused_output.jpg', fused_img)
    print("Fusion complete, image saved as 'fused_output.jpg'")
else:
    print("Fusion failed.")


