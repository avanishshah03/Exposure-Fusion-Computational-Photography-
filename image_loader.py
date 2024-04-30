import cv2

def load_image(file_path):
    # Load an image from file
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image from {file_path}")
    else:
        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, width, height):
    # Resize image to specified width and height
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
