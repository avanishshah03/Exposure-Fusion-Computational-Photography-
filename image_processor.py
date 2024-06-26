import cv2
import numpy as np
from image_loader import resize_image

class ImageProcessor:
    def __init__(self):
        self.aligner = cv2.createAlignMTB()
        self.merger = cv2.createMergeMertens()

    def align_images(self, images, width, height):
        resized_images = [resize_image(img, width, height) for img in images if img is not None]
        self.aligner.process(resized_images, resized_images)
        return resized_images

    def merge_images(self, images):
        images = [img.astype(np.float32) / 255.0 for img in images]
        fused_image = self.merger.process(images)
        # normalizing here to convert the image back to 8-bit format
        return cv2.normalize(fused_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
