% Drew Afromsky 

% Adapted/Modified from: Investigate Network Predictions Using Class Activation
% Mapping and Deep Learning Visualizations: CAM Visualization 2

% CAMshow overlays the class activation map on a darkened, grayscale version of the image 'test'. 
% The function resizes the class activation map to the size of 'test', normalizes it, thresholds it from below, 
% and visualizes it using a jet colormap.

function CAMshow(test,CAM)
  imSize = size(test);
  CAM = imresize(CAM,imSize(1:2));
  
  CAM = normalizeimage(CAM);
  CAM(CAM < .2) = 0;
  cmap = jet(255).*linspace(0,1,255)';
 
  CAM = ind2rgb(uint8(CAM*255),cmap)*255;

  combinedImage = double(rgb2gray(test))/2 + CAM;
  combinedImage = normalizeimage(combinedImage)*255;
  imshow(uint8(combinedImage));
end

