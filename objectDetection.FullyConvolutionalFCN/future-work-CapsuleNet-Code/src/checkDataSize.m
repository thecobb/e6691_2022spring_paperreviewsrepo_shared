function [] = checkDataSize(imds)
%Check if all provided training images are in RGB format
imagesPath = [imds.Files];
for i=1:numel(imagesPath)
   currentImage = readimage(imds, i);
   [~, ~, colorChannels] = size(currentImage);
   if colorChannels == 1 || colorChannels == 2
       newImage = cat(3, currentImage, currentImage, currentImage);
       imwrite(newImage, imagesPath{i});
%        imwrite(newImage, strcat(imagesPath{i},'.jpg'));
       fprintf("Found and overwrote image with a single color channel in path %s \n", imagesPath{i});
   end
end
end