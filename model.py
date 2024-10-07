import os
import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
from config import MODEL_PATH

class AODNet(nn.Module):
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
	
	# Define the weight initialization function to ensure the network starts training with appropriately set parameters
	def _initialize_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.normal_(module.weight, 0.0, 0.02)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.normal_(module.weight, 1.0, 0.02)
				nn.init.constant_(module.bias, 0)


def load_model():
	# Load your model here
    model = AODNet()
    model.eval()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=torch.device('cpu')))
    return model

model = load_model()  # Load model (can be done outside this function for efficiency)


def dehaze_image(image):
    data_hazy = Image.open(image).convert("RGB")  # Ensure the image is RGB
    data_hazy = np.asarray(data_hazy)/255.0  # Normalize to [0, 1]
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)  # Change to (C, H, W)
    data_hazy = data_hazy.unsqueeze(0)  # Add batch dimension

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        clean = model(data_hazy)  # Forward pass through the model

    clean_image = clean[0].detach().permute(1, 2, 0).numpy()
    clean_image = (clean_image * 255).astype(np.uint8)  # Scale back to [0, 255]
    
    return clean_image  # Return the cleaned image