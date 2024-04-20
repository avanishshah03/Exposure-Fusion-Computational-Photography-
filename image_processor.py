import cv2
import numpy as np
from image_loader import resize_image

class ImageProcessor:
    def __init__(self):
        self.aligner = cv2.createAlignMTB()
        self.merger = cv2.createMergeMertens()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def align_images(self, images, width, height):
        resized_images = [resize_image(img, width, height) for img in images if img is not None]
        self.aligner.process(resized_images, resized_images)
        return resized_images

    def merge_images(self, images):
        images = [img.astype(np.float32) / 255.0 for img in images]
        fused_image = self.merger.process(images)
        return cv2.normalize(fused_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def enhance_local_contrast(self, image):
        channels = cv2.split(image)
        enhanced_channels = [self.clahe.apply(ch) for ch in channels]
        return cv2.merge(enhanced_channels)
