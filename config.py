"""
Configuration constants and settings for the Image Dehazing application.

This module centralizes all configuration values including file paths,
upload constraints, and application settings.
"""
import os

# Base directory configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# File path constants
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'dehazer_epoch_9.pth')

# File type validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/jpg'}

# File upload size constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload size for 

# Image dimension constraints
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# File retention settings
FILE_RETENTION_TIME = 24 * 60 * 60  # 24 hours in seconds

# Security settings
SECRET_KEY = os.environ.get('SECRET_KEY', None)
if SECRET_KEY is None:
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    import warnings
    warnings.warn(
        'SECRET_KEY not set in environment. Using generated key. '
        'Set SECRET_KEY environment variable for production.',
        UserWarning
    )

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

