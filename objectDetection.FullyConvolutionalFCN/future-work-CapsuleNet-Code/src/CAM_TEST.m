%% Authored by Drew Afromsky and Jacob Nye
% May 1, 2019

clc
clear all
net = load('Uptodate_CapsuleSegnet.mat'); % Load the network

%%
% Move to folder 'New Folder'
fn = 'C:\Users\drewa\Documents\MATLAB\3-Feature Visualization-20190502T025640Z-001\3-Feature Visualization\New Folder';
ds=imageDatastore(fn); % Create image datastore with all of the images from the data set
filenames = ds.Files; % List of file names in datastore
imgs = readall(ds); % read in the images from the datastore
%%
for j=1:length(imgs)
test=imgs{j}; % All the images from 'imgs' read into a cell

% Return the activations from the ReLU layer following the last
% convolutional layer ('decoder1_relu_2')
imageActivations = activations(net.CapsuleSegNet5,test,'decoder1_relu_2'); 

scores = squeeze(mean(imageActivations,[1 2])); % Calculate the scores of activations per class
[~,classIds] = maxk(scores,3); % Extract class IDs
classActivationMap = imageActivations(:,:,classIds(1)); % Generate class activation map

% Calculate the top class labels and the final normalized class scores.
scores = exp(scores)/sum(exp(scores));
maxScores = scores(classIds);
labels = net.CapsuleSegNet5.Layers(91, 1).Classes;
% Visualize class activation mapping for every image in the dataset
        figure;
        name = filenames{j}(113:end);
        CAMshow(test,classActivationMap); 
        title(string(labels) + ", " + string(maxScores));
        drawnow;
        caIm = getframe();
        imwrite(caIm.cdata,name)
        close all
end
