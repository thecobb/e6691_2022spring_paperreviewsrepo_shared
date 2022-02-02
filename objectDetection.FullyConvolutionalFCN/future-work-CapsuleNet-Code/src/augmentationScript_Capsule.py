# Jacob Nye
# BMEN 4000 Final
# 4/29/2019

import Augmentor

p = Augmentor.Pipeline("C:\\Users\\JacobNye\\Documents\\Preprocessing\\wcetraining\\Training2\\TrainImages")
p.ground_truth("C:\\Users\\JacobNye\\Documents\\Preprocessing\\wcetraining\\Training2\\TrainLabels")

# p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=1)
p.flip_top_bottom(probability=1)

# p.rotate90(probability=0.0)
# p.rotate270(probability=0.5)



# p.shear(probability=1,max_shear_left=8,max_shear_right =8)
# p.skew(probability=1,magnitude=0.4)

# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)


p.process()

print('Done')

