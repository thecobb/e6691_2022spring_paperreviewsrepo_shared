import os
from shutil import copy
import numpy as np
from PIL import Image

# destination = "C:\\Users\\JacobNye\\Documents\\MATLAB\\Zuker_Lab\\FPdata\\Jcopy-181204\\Behavior Notes\\polypAll\\changed"
# target = "C:\\Users\\JacobNye\\Documents\\MATLAB\\Zuker_Lab\\FPdata\\Jcopy-181204\\Behavior Notes\\polypAll\\augment_LeftRight"

# destination = "C:\\Users\\JacobNye\\Documents\\MATLAB\\Zuker_Lab\\FPdata\\Jcopy-181204\\Behavior Notes\\polypAll\\output"
target= "C:\\Users\\JacobNye\\Documents\\Preprocessing\\wcetraining\\wcetraining\\Cleaned Data\\normal_images"
# destination = "C:\\Users\\JacobNye\\Documents\\Preprocessing\\wcetraining\\wcetraining\\inflammatory_masks"

#destination_list = os.listdir(destination)
#data_dir_list = os.listdir(target)

# zero_image = np.zeros(576,576)

for file in os.listdir(target):
      # if ".tif" in file:
    fname = str(file)
    img = Image.new('RGB', (576, 576))
    img.save(fname, "JPEG")

print('Done')

#_groundtruth_(1)_polypAll_1-1.png_b96ced0d-8ede-4123-a8f0-25cb84057552