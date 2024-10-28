We obtained our dataset from the NYU2 dataset (Silberman et al., 2012), which includes approximately 1,500 ground truth images in the folder "ori_images" and around 27,000 synthetically hazed images in the folder "hazy_images." 
Each image in "ori_images" is named in the format “NYU2_x.jpg,” where x is an integer. 
For each ground truth image, multiple hazed versions are provided, with filenames formatted as “NYU2_x_y_z,” where y and z vary to indicate different haze levels. 
This dataset provided a comprehensive range of hazy conditions, supporting effective model training for diverse scenarios.

Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). Indoor segmentation and support inference from RGBD images. In A. Fitzgibbon, S. Lazebnik, P. Perona, Y. Sato, & C. Schmid (Eds.), Computer vision – ECCV 2012 (Vol. 7576, pp. 346-360). Springer. https://doi.org/10.1007/978-3-642-33715-4_54