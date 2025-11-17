"""
Streamlit application for image dehazing using AOD-Net.

This is the main entry point for the image dehazing application.
Users can upload hazy images and receive dehazed results.
"""
import streamlit as st
from PIL import Image
import io
import tempfile
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the existing model dehazing function and config
try:
    from model import dehaze_image
    from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT
    MODEL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import model or config: {e}")
    dehaze_image = None
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB fallback
    MAX_IMAGE_WIDTH = 4096
    MAX_IMAGE_HEIGHT = 4096
    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100
    MODEL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Image Dehazing | AOD-Net",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_dimensions(image_path: str) -> tuple[bool, str]:
    """
    Validate image dimensions.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return False, f"Image too small. Minimum: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px"
            
            if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
                return False, f"Image too large. Maximum: {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}px"
            
            return True, None
    except Exception as e:
        return False, f"Unable to read image: {str(e)}"


# Main header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("✨ Image Dehazing with AOD-Net")
st.markdown("Remove atmospheric haze from your images using advanced AI technology")
st.markdown('</div>', unsafe_allow_html=True)

# Check if model is available
if not MODEL_AVAILABLE:
    st.error("⚠️ **Model not available** - Please ensure `model.py` and `config.py` are properly configured and dependencies are installed.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file to dehaze",
    type=list(ALLOWED_EXTENSIONS),
    accept_multiple_files=False,
    help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS).upper()}. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB"
)

if uploaded_file is None:
    st.info(f"👆 **Upload an image** to get started!\n\n- Supported formats: {', '.join(ALLOWED_EXTENSIONS).upper()}\n- Maximum file size: {MAX_FILE_SIZE / (1024*1024):.0f} MB\n- Image dimensions: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px to {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}px")
else:
    # Validate file
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    
    # File size check
    if file_size_mb * 1024 * 1024 > MAX_FILE_SIZE:
        st.error(f"❌ **File too large** - Maximum allowed size is {MAX_FILE_SIZE / (1024*1024):.0f} MB. Your file is {file_size_mb:.2f} MB.")
    elif not allowed_file(uploaded_file.name):
        st.error(f"❌ **Invalid file type** - Please upload a {', '.join(ALLOWED_EXTENSIONS).upper()} image.")
    else:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Validate image dimensions
            is_valid, error_msg = validate_image_dimensions(tmp_path)
            if not is_valid:
                st.error(f"❌ **Image validation failed**: {error_msg}")
            else:
                # Process the image
                with st.spinner("🔄 Processing your image with AOD-Net model... This may take a few seconds..."):
                    try:
                        result = dehaze_image(tmp_path)
                    except ValueError as e:
                        st.error(f"❌ **Validation Error**: {str(e)}")
                        result = None
                    except RuntimeError as e:
                        error_msg = str(e)
                        if 'OutOfMemoryError' in error_msg or 'memory' in error_msg.lower():
                            st.error("❌ **Memory Error**: Image is too large to process. Please try a smaller image.")
                        else:
                            st.error(f"❌ **Processing Error**: {error_msg}")
                        result = None
                    except Exception as e:
                        st.error(f"❌ **Unexpected Error**: {str(e)}")
                        logger.exception("Error during dehazing")
                        result = None

                # Display results
                if result is not None and isinstance(result, np.ndarray):
                    original_image = Image.open(tmp_path).convert("RGB")
                    dehazed_image = Image.fromarray(result)

                    st.success("✅ **Dehazing complete!** Compare the results below.")
                    
                    # Side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📷 Original Image")
                        st.image(original_image, use_container_width=True)
                        st.caption(f"Dimensions: {original_image.size[0]} × {original_image.size[1]} pixels")
                    
                    with col2:
                        st.subheader("✨ Dehazed Image")
                        st.image(dehazed_image, use_container_width=True)
                        st.caption(f"Dimensions: {dehazed_image.size[0]} × {dehazed_image.size[1]} pixels")

                    # Download button
                    st.markdown("---")
                    buf = io.BytesIO()
                    # Preserve original format if possible
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG'}
                    save_format = format_map.get(file_ext, 'PNG')
                    dehazed_image.save(buf, format=save_format)
                    buf.seek(0)
                    
                    st.download_button(
                        "⬇️ Download Dehazed Image",
                        data=buf,
                        file_name=f"dehazed_{os.path.basename(uploaded_file.name)}",
                        mime=f"image/{save_format.lower()}",
                        use_container_width=True
                    )

                elif result is not None:
                    st.error("❌ **Unexpected output format** - Expected a numpy array (H, W, C).")

        except Exception as e:
            st.exception(e)
            logger.exception("Unexpected error during processing")
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses **AOD-Net** (All-in-One Dehazing Network) to remove 
    atmospheric haze from images while preserving fine details.
    
    **How it works:**
    1. Upload a hazy image
    2. The AI model processes it
    3. Download your clear, dehazed result
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - **Large images** may take longer to process
    - **Memory usage** increases with image size
    - For best results, use images with visible haze
    - Supported formats: PNG, JPG, JPEG
    """)
    
    st.header("⚙️ Technical Details")
    st.markdown(f"""
    - **Model**: AOD-Net (All-in-One Dehazing Network)
    - **Max file size**: {MAX_FILE_SIZE / (1024*1024):.0f} MB
    - **Image dimensions**: {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT}px to {MAX_IMAGE_WIDTH}×{MAX_IMAGE_HEIGHT}px
    - **Supported formats**: {', '.join(ALLOWED_EXTENSIONS).upper()}
    """)
    
    st.markdown("---")
    st.markdown("**Powered by AOD-Net**")
