%% BMEN 4000 Final Project: Wireless Capsule Endoscopy Lesion Detection
% Jacob Nye
% 
% This script sets up the neural network for performing semantic segmentation 
% on capsule endoscopy images from the MICCAI 2018 GIANA WCE Challenge. There 
% are 3 classes for the network to classify: inflammatory, vascular and healthy. 
% In the ground truth mask images, white pixels delineate inflammatory lesions, 
% vascular lesions, and healthy lesions. This code was modified to train the multiple 
% iterations of each neural network type. The best performances came from the 
% Segnet, DeepLabV3+, and FCN-8 networks. We tried multiple iterations of training 
% U-Net models but the training was intractable and training performance oscillated 
% around 50-60%.
% 
% % Load the data and split into training, validation, and test sets.
% In this section, the 1812 images are loaded into image datastores and split 
% into training, validation, and test sets, with an 85% training data, 10% validation, 
% and 5% test data.

% dataSetDir = 'C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019-20190418T194029Z-001\CleanedData_Alicia_17Apr2019';
% The one below was used for augmented data and was swithced on 5/3 for
% Unet dataset
% dataSetDir = 'C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019';

% Unet dataset 512x512 preaugmentation
dataSetDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2';
imageDir = fullfile(dataSetDir,'Images');
labelDir = fullfile(dataSetDir,'Labels');
%%
% Augmented
% ************* Use this for augmented data ******************


% dataSetDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\AugmDir'
% dataSetDir = 'C:\Users\JacobNye\Documents\Preprocessing\TrainData';
% imageDir = fullfile(dataSetDir,'Images');
% labelDir = fullfile(dataSetDir,'Labels');
% imageDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TrainImagesAug';
% trainLabel = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TrainLabelsAug';

% Temp sanity check - use non augmented data
imageDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TrainImages';
trainLabel = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TrainLabels'


% For second attempt at augmenting data (this time without data leak)
% Load in test and validation data explicitly
% Load in testing and validation image datastores
testDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TestImages';
testLabel = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\TestLabels';
ValDir = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\ValidationImages';
ValLabel = 'C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\ValidationLabels';
%%
% Continue to set up
classNames = [
    "Inflammatory"
    "Vascular"
    "Normal"
    ]; 
labelIDs   = { ...
    [
    255 255 255; ... %   inflammatory
    ]
    [
    125 125 125; ...%Vascular 
    ]
    [
    000 000 000; ...%healthy
    ]
    };
    
    
    
imds = imageDatastore(imageDir);
% inputsize = [256 256 3];
% Load image datastores for augmented data and test/val sets

% Uncomment for augmented
imdsTrain = imageDatastore(imageDir);
imdsTest = imageDatastore(testDir);
imdsVal = imageDatastore(ValDir);

% auimds = augmentedImageDatastore(inputsize,imds,'ColorPreprocessing',"rgb2gray")
% patchSize = [256,256];
% patchds = randomPatchExtractionDatastore(imds,pxds,patchSize,'PatchesPerImage',32)


% ******************************** For original uncomment This
% pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
% *****************************************************************

% If switching, uncomment below
pxdsTrain = pixelLabelDatastore(trainLabel,classNames,labelIDs);
pxdsTest = pixelLabelDatastore(testLabel,classNames,labelIDs);
pxdsVal = pixelLabelDatastore(ValLabel,classNames,labelIDs);


% ******************************************************************
% Partition data by randomly selecting 80% of the data for training. The
% rest is used for testing.
    
% % Set initial random state for example reproducibility.
% rng(0); 
% numFiles = numel(imds.Files);
% shuffledIndices = randperm(numFiles);
% 
% % % Use 60% of the images for training.
% numTrain = round(0.85 * numFiles);
% trainingIdx = shuffledIndices(1:numTrain);
% 
% % % Use 10% of the images for validation
% numVal = round(0.10 * numFiles);
% valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
% 
% % % Use the rest for testing.
% testIdx = shuffledIndices(numTrain+numVal+1:end);
% 
% % % Create image datastores for training and test.
% trainingImages = imds.Files(trainingIdx);
% valImages = imds.Files(valIdx);
% testImages = imds.Files(testIdx);

% % Uncomment for augmented
trainingImages = imdsTrain.Files;
valImages = imdsVal.Files;
testImages = imdsTest.Files;


imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% trainingLabels = pxds.Files(trainingIdx);
% valLabels = pxds.Files(testIdx);
% testLabels = pxds.Files(valIdx);

% imdsTrainLab = imageDatastore(trainingLabels);
% imdsValLab = imageDatastore(valLabels);
% imdsTestLab = imageDatastore(testLabels);



% uncomment for train
% Extract class and label IDs info.
classes = pxdsTrain.ClassNames;

% % Uncomment for augmented
% % Create pixel label datastores for training and test.
trainingLabels = pxdsTrain.Files;
valLabels = pxdsVal.Files;
testLabels = pxdsTest.Files;
% 
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);

% gTruthCapsule = groundTruth(imdsTrain,)
%%
% Create image augmenter to augment images
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],"RandRotation",[-20 20]);
%%
% % With augmentation
%  pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
%     'DataAugmentation',augmenter);

% % No on the fly augmentation
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
%% Use this before the pixel classification layers

% Visualization of class pixel count over whole dataset
%Do
resnet18();
% tbl = countEachLabel(pxds)
tbl = countEachLabel(pxdsTrain)

frequency = tbl.PixelCount/sum(tbl.PixelCount);
% Plot frezquency of class out of 100% of pixels
figure;
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%%
% Create DeppLabV3+
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [509 495 3];
% imageSize = [512 512 3];

% Specify the number of classes.
numClasses = numel(classes);

% Create DeepLab v3+.   
lgraph = helperDeeplabv3PlusResnet18(imageSize, numClasses);
%%
% Correct the unbalanced classes using class-weighted cross entropy pixel
% classification layer
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;

%  Alternative  weighting scheme
% totalNumberOfPixels = sum(tbl.PixelCount);
% frequency = tbl.PixelCount / totalNumberOfPixels;
% classWeights = 1./frequency

classWeights = median(imageFreq) ./ imageFreq
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);

% Alternative dicePixelClassificationLayer for training UNet
% pxLayer = dicePixelClassificationLayer('labels');

% ConvLayer = convolution2dLayer(1,3,'Name','conv_end');
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% lgraph = addLayers(lgraph,ConvLayer);
% lgraph = disconnectLayers(lgraph,'dec_crop2','softmax-out');
% newlgraph = connectLayers(lgraph,'dec_crop2','conv_end');
% newlgraph = connectLayers(newlgraph,'conv_end','softmax-out');

analyzeNetwork(lgraph)
% disp('done')
%% 
% *Alternative training options, sometimes used.*

% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);
l2reg = 0.05;
% minibatchSize = 16;

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',8,...
    'Momentum',0.9, ...
    'LearnRateDropFactor',0.1,...
    'InitialLearnRate',0.001, ...
    'ValidationData',pximdsVal,...
    'ValidationFrequency',1152,...
    'L2Regularization',l2reg, ...
    'GradientThreshold',0.05, ...
    'L2Regularization',l2reg,...
   'GradientThresholdMethod','l2norm',...
    'MaxEpochs',20, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','multi-gpu',...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',5,...
    'Plots','training-progress',...
    'ValidationPatience', 4); ...
        
%%
%%Start Training
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
%% Train a U-net

% Create U-Net.

imageSize = [512 512 3];
numClasses = 3;
lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',4);

inputTileSize = [512,512,3];
lgraph = createUnet(inputTileSize);

analyzeNetwork(lgraph)
%% Train a FCN

imageSize = [509 495];
numClasses = 3;
lgraph = fcnLayers(imageSize,numClasses)
% analyzeNetwork(lgraph)
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"pixelLabels",pxLayer);

%% Train a segnet
% Just do it

% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [509 495 3];

% Specify the number of classes.
numClasses = numel(classes);
% Use a model VGG16 or VGG19
lgraph = segnetLayers(imageSize,numClasses,'vgg16')
% analyzeNetwork(lgraph)
%%
pximdsTrain = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
tbl_seg = countEachLabel(pximdsTrain);

numberPixels = sum(tbl_seg.PixelCount);

frequency_seg = tbl_seg.PixelCount / numberPixels;
% classWeights = 1 ./ frequency_seg;
classWeights = median(frequency_seg) ./ frequency_seg;


% imageFreq1 = tbl_seg.PixelCount ./ tbl_seg.ImagePixelCount;
% classWeights1 = median(imageFreq1) ./ imageFreq1


pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl_seg.Name,'ClassWeights',classWeights);

% Deep Lab
% lgraph = replaceLayer(lgraph,"pixelLabels",pxLayer);
% Unet
lgraph = replaceLayer(lgraph,"Segmentation-Layer",pxLayer);

% lgraph = removeLayers(lgraph,'pixelLabels');
% lgraph = addLayers(lgraph, pxLayer);
% DeepLabV3
% lgraph = connectLayers(lgraph,'softmax','labels')
% Unet
% lgraph = connectLayers(lgraph,'softmax','labels')

%Say it's done
disp('done')

%% *Network training options*

% Define validation data.

% % Uncomment below when needed
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'LearnRateSchedule','piecewise',...
    'InitialLearnRate',1e-3, ...
    'ValidationData',pximdsVal,...
    'ValidationFrequency',61,...
    'MaxEpochs',10, ...    
    'LearnRateDropFactor',0.3,...
    'LearnRateDropPeriod',5,...
    'L2Regularization',0.001, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',5,...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'multi-gpu',...
    'ValidationPatience', 4); ...
%% Visualizing the class distribution (on a pixel level) of the train, validation, and testing datasets

%Training Distribution - Class Visualization

tbl_train = countEachLabel(pxdsTrain)
frequency_train = tbl_train.PixelCount/sum(tbl_train.PixelCount);
% Plot frequency of class out of 100% of pixels
bar(1:numel(classes),frequency_train)
xticks(1:numel(classes)) 
xticklabels(tbl_train.Name)
xtickangle(45)
ylabel('Frequency')
%%
%Validation Distribution - Class Visualization

tbl_val = countEachLabel(pxdsVal)
frequency_val = tbl_val.PixelCount/sum(tbl_val.PixelCount);
% Plot frequency of class out of 100% of pixels
bar(1:numel(classes),frequency_val)
xticks(1:numel(classes)) 
xticklabels(tbl_val.Name)
xtickangle(45)
ylabel('Frequency')
%%
%Test Distribution - Class Visualization

tbl_test = countEachLabel(pxdsTest)
frequency_test = tbl_test.PixelCount/sum(tbl_test.PixelCount);
% Plot frequency of class out of 100% of pixels
bar(1:numel(classes),frequency_test)
xticks(1:numel(classes)) 
xticklabels(tbl_test.Name)
xtickangle(45)
ylabel('Frequency')
%% Exploring the best and worst predictions of the network and evaluating  the network on the test and validation data.

% Signle image Test
% Use this code block to test the network on a few different images

% I_test = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U112.jpg');
I_test = imread('C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\Images\EI4.jpg');
% I_test  = imresize(I_test,[256 256]);
[C,scores] = semanticseg(I_test,CapsuleNet12);
B = labeloverlay(I_test,C,'Transparency',0.50);
figure;
imshow(B)
title('AGD105')

cMapCap = capsuleColorMap();
pixelLabelColorbar1(cMapCap,classNames)

% I_test_mask =  imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\U112.jpg');
I_test_mask = imread('C:\Users\JacobNye\Documents\Preprocessing\wcetraining\Training2\Labels\EI4.jpg');
figure;
imshow(I_test_mask)
cMapCap = capsuleColorMap();
pixelLabelColorbar1(cMapCap,classNames)
figure;
imagesc(scores)
colorbar()
figure;
imshow(I_test)
%%
%Extract boundary using thresholded scores
BW_test = imbinarize(scores,0.55);
BW2 = ~BW_test;
imshow(BW2)
B2 = labeloverlay(I_test,BW2,'Transparency',0.2);
imshow(B2)
%%
% Validation Set - larger than test
pxdsResults_Test = semanticseg(imdsTest,CapsuleNet12, ...
    'MiniBatchSize',8, ...
    'WriteLocation',tempdir, ...
    'Verbose',true);

metrics_Test = evaluateSemanticSegmentation(pxdsResults_Test,pxdsTest);
%%
% Evaluate on Validation data
pxdsResults_val = semanticseg(imdsVal,CapsuleNet12,...
    'MiniBatchSize',8, ... 
    'WriteLocation',tempdir);
metrics_Val = evaluateSemanticSegmentation(pxdsResults_val,pxdsVal);
%%
metrics_Test.DataSetMetrics
metrics_Test.ClassMetrics
%%
metrics_Val.DataSetMetrics
metrics_Val.ClassMetrics
%% Explore the best and worst images from the network

% After finding worst accuracy image in excel, plot it

% Worst Image 
% I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U498.jpg');
I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U139.jpg');
[C_bad] = semanticseg(I_bad,CapsuleNet2)
B_bad = labeloverlay(I_bad,C_bad,'Transparency',0.55);
subplot(3,1,1)
imshow(B_bad)
title('Worst Image - CapsuleNetV0.2 - image U139 - Mean Accuracy: 33%')

cBadMapCap = capsuleColorMap();
pixelLabelColorbar1(cBadMapCap,classNames)
subplot(3,1,2)
I_bad_label = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\U139.jpg');
imshow(I_bad_label)
subplot(3,1,3)
imshow(I_bad)
%%
% Bad Image 2
% I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U169.jpg');
I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U591.jpg')
[C_bad] = semanticseg(I_bad,CapsuleNet2)
B_bad = labeloverlay(I_bad,C_bad,'Transparency',0.55);
subplot(3,1,1)
imshow(B_bad)
title('2nd Worst Image - CapsuleNetV0.2 - image U591 - Mean Accuracy: 46%')
cBadMapCap = capsuleColorMap();
pixelLabelColorbar1(cBadMapCap,classNames)
subplot(3,1,2)
I_bad_label = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\U169.jpg');
imshow(I_bad_label)
subplot(3,1,3)
imshow(I_bad)
%%
% Bad image 3
% I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\EI3.jpg');
% I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U174.jpg');
I_bad = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\AGD44.jpg');
[C_bad] = semanticseg(I_bad,CapsuleNet2)
B_bad = labeloverlay(I_bad,C_bad,'Transparency',0.55);
subplot(3,1,1)
imshow(B_bad)
cBadMapCap = capsuleColorMap();
pixelLabelColorbar1(cBadMapCap,classNames)
title('3rd Worst Image - image AGD44 - Mean Accuracy: 48.5%')
subplot(3,1,2)
I_bad_label = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\AGD44.jpg');
imshow(I_bad_label)
subplot(3,1,3)
imshow(I_bad)

%%
% Best image 1
% I_good = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\AGD869.jpg');
I_good = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\AGD1098.jpg');
[C_good] = semanticseg(I_good,CapsuleNet2)
B_good = labeloverlay(I_good,C_good,'Transparency',0.55);
subplot(3,1,1)
imshow(B_good)
cGoodMapCap = capsuleColorMap();
pixelLabelColorbar1(cGoodMapCap,classNames)
title('1st Best Image - image AGD1098 - Mean Accuracy: 99.76%')
subplot(3,1,2)
I_good_label = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\AGD1098.jpg');
imshow(I_good_label)
subplot(3,1,3)
imshow(I_good)

%%
% Best image 2
% I_good = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U557.jpg');
% I_good = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\AGD694.jpg');
I_good = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Images\U39.jpg');

[C_good] = semanticseg(I_good,CapsuleNet7)
B_good = labeloverlay(I_good,C_good,'Transparency',0.55);
subplot(3,1,1)
imshow(B_good)
cGoodMapCap = capsuleColorMap();
pixelLabelColorbar1(cGoodMapCap,classNames)
title('Best Inflammatory Image - image U39 - Mean Accuracy: 98.04%')
subplot(3,1,2)
I_good_label = imread('C:\Users\JacobNye\Downloads\Compressed\CleanedData_Alicia_17Apr2019\CleanedData_Alicia_17Apr2019\Labels\U39.jpg');
imshow(I_good_label)
subplot(3,1,3)
imshow(I_good)