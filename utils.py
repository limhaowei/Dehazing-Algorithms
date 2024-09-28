import os
from PIL import Image
from config import ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Implement your image preprocessing logic here
    # This is a placeholder implementation
    return image

def postprocess_image(image):
    # Implement your image postprocessing logic here
    # This is a placeholder implementation
    return image



