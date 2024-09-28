import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'your_model_file.h5')

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

