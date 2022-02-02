function [] = separateTrainData(imds,newFilePathPart)
%Copy all images into a new folder 
imagesPath = [imds.Files];
for i=1:numel(imagesPath)
%    currentImage = readimage(imds, i);
   currentImage = imread(imagesPath{i});
   [fPath, fName, fExt] = fileparts(imagesPath{i});
%    disp(fName)
   newFile = fullfile(newFilePathPart,strcat(fName,'.jpg'));
   imwrite(currentImage, newFile);
  
end
disp('File copying complete')
end
