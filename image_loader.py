import cv2

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image from {file_path}")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
