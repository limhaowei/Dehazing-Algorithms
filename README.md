# Dehazing-Algorithms

### Guide on Training the model 
To train the dehazing model (AOD-Net), follow these steps:
1.  Download the Google Colab Notebook.
2.  Set Up Your Environment: Make sure your environment has the necessary libraries installed. You can do this by running the following commands in a Google Colab cell:
!pip install torch torchvision scikit-image opencv-python
3.  Upload and Organize Your Dataset: 
Your dataset consists of two folders:
ori_images/: ~1500 ground truth images (e.g., “NYU2_x.jpg”).
hazy_images/: ~27K hazy images (e.g., “NYU2_x_y_z”)
Make sure the dataset is organized as described above, and upload the images to Google Colab or link them from a cloud storage service like Google Drive.
4.  Set the Model Configuration:
In the Args class, configure the paths to your dataset, learning rate, weight decay, and other training parameters.
5.  Data Preparation:
The DehazeDataManager class handles loading and splitting the dataset into training, validation, and test sets. It loads hazy and ground truth images from the paths you specify.
6.  Model Definition:
The AODNet class is your dehazing model. It is initialized and transferred to the GPU for training.
7.  Start the Training Process:
In the BaseTrainer class, the fit method trains the model over multiple epochs. This method will save checkpoints after each epoch, and you can use these checkpoints to evaluate the model later.
8.  Evaluate on the Test Set:
After training, the model is evaluated on the test set for each epoch. The evaluate method computes the SSIM and FSIM metrics, and the model that gives the best results is saved.
9.  Save the Trained Model:
Your model checkpoints are saved in the snapshots folder. After training, you can load the best model and use it for further evaluation or inference.