"""
AOD-Net model implementation for image dehazing.

This module contains the AOD-Net (All-in-One Dehazing Network) architecture
and functions for loading the model and processing hazy images to remove
atmospheric haze.
"""
import os
import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
import logging
from config import MODEL_PATH

# Configure logging
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading the model on every request
_model_cache = None
_device = None


class AODNet(nn.Module):
	"""
	AOD-Net: All-in-One Dehazing Network
	A lightweight CNN architecture for image dehazing that removes atmospheric haze
	from images while preserving fine details.
	"""
	def __init__(self):
		super(AODNet, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
		self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
		self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
		self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

		# Apply weight initialization
		self._initialize_weights()

	def forward(self, x):
		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		concat1 = torch.cat((x1,x2), 1)
		x3 = self.relu(self.e_conv3(concat1))
		concat2 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv4(concat2))
		concat3 = torch.cat((x1,x2,x3,x4),1)
		x5 = self.relu(self.e_conv5(concat3))
		clean_image = self.relu((x5 * x) - x5 + 1)
		return clean_image
	
	def _initialize_weights(self):
		"""
		Initialize network weights for stable training.
		
		Uses normal distribution initialization for convolutional layers
		and batch normalization layers to ensure stable gradient flow
		during training.
		"""
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.normal_(module.weight, 0.0, 0.02)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.normal_(module.weight, 1.0, 0.02)
				nn.init.constant_(module.bias, 0)


def get_device():
	"""
	Detect and return the best available device (GPU or CPU).
	Caches the device for efficiency.
	"""
	global _device
	if _device is not None:
		return _device
	
	if torch.cuda.is_available():
		_device = torch.device('cuda')
		logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
	else:
		_device = torch.device('cpu')
		logger.info('Using CPU')
	
	return _device


def load_model():
	"""
	Lazy load and cache the model for efficiency.
	Returns the model in evaluation mode.
	"""
	global _model_cache
	
	# Return cached model if available
	if _model_cache is not None:
		return _model_cache
	
	try:
		device = get_device()
		logger.info(f'Loading model from: {MODEL_PATH}')
		
		# Check if model file exists
		if not os.path.exists(MODEL_PATH):
			raise FileNotFoundError(f'Model weights not found at {MODEL_PATH}')
		
		# Load model architecture and weights
		model = AODNet()
		model.load_state_dict(
			torch.load(MODEL_PATH, weights_only=True, map_location=device)
		)
		
		# Move to device and set evaluation mode
		model = model.to(device)
		model.eval()
		
		# Cache the model
		_model_cache = model
		logger.info('Model loaded successfully')
		return model
	
	except Exception as e:
		logger.error(f'Failed to load model: {str(e)}', exc_info=True)
		raise RuntimeError(f'Model loading error: {str(e)}')


def dehaze_image(image_path):
	"""
	Dehaze an image using the AOD-Net model.
	
	Args:
		image_path (str): Path to the input hazy image
	
	Returns:
		np.ndarray: Dehazed image as RGB numpy array (0-255 range)
	
	Raises:
		ValueError: If image cannot be loaded or processed
		RuntimeError: If model inference fails
	"""
	try:
		device = get_device()
		model = load_model()
		
		# Validate image path
		if not os.path.exists(image_path):
			raise ValueError(f'Image file not found: {image_path}')
		
		try:
			# Load and prepare image
			logger.info(f'Loading image: {image_path}')
			data_hazy = Image.open(image_path).convert("RGB")
		except Exception as e:
			raise ValueError(f'Unable to open image file: {str(e)}')
		
		# Normalize image to [0, 1]
		data_hazy = np.asarray(data_hazy) / 255.0
		
		# Convert to tensor and permute to (C, H, W)
		data_hazy = torch.from_numpy(data_hazy).float()
		data_hazy = data_hazy.permute(2, 0, 1)
		
		# Add batch dimension
		data_hazy = data_hazy.unsqueeze(0)
		
		# Move to device
		data_hazy = data_hazy.to(device)
		
		# Inference
		with torch.no_grad():
			clean = model(data_hazy)
		
		# Convert output to numpy array
		clean_image = clean[0].detach().cpu().permute(1, 2, 0).numpy()
		
		# Clip values to [0, 1] and scale to [0, 255]
		clean_image = np.clip(clean_image, 0, 1)
		clean_image = (clean_image * 255).astype(np.uint8)
		
		logger.info(f'Image successfully dehazed: {image_path}')
		return clean_image
	
	except (ValueError, FileNotFoundError) as e:
		logger.error(f'Image validation error: {str(e)}')
		raise ValueError(f'Invalid image file: {str(e)}')
	except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
		logger.error(f'Model inference error: {str(e)}', exc_info=True)
		raise RuntimeError(f'Failed to process image with model: {str(e)}')
	except (OSError, IOError) as e:
		logger.error(f'File I/O error during dehazing: {str(e)}', exc_info=True)
		raise RuntimeError(f'Unable to read or write image file: {str(e)}')
	except Exception as e:
		logger.error(f'Unexpected error during dehazing: {str(e)}', exc_info=True)
		raise RuntimeError(f'Image processing failed: {str(e)}')