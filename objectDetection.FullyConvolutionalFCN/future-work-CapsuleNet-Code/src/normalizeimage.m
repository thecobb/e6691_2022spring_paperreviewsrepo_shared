% Drew Afromsky 

% Adapted/Modified from: Investigate Network Predictions Using Class Activation
% Mapping and Deep Learning Visualizations: CAM Visualization 2

% This function simply normalizes the input image

function N = normalizeimage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end
