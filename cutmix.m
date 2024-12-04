clear all
warning off

% dataset identification number
datas = 29;

% load the dataset
load(strcat('DatasColor_', int2str(datas)), 'DATA');
NF = size(DATA{3}, 1); % number of folds for cross-validation
DIV = DATA{3}; % division between training and test sets
DIM1 = DATA{4}; % number of training patterns
DIM2 = DATA{5}; % number of patterns
yE = DATA{2}; % labels for the patterns
NX = DATA{1}; % images

% define category names for the classes
categoryNames = {'CLL', 'FL', 'MCL'};

% convert numeric labels to categorical using the predefined category names
yE = categorical(yE, 1:3, categoryNames);

% load the pre-trained AlexNet model
net = alexnet;  %  for image classification
siz = [227 227]; % image size for input to AlexNet
% VGG-16 model:
% net = vgg16;
% siz = [224 224];

% set the parameters for NN training
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim = 'sgdm';
options = trainingOptions(metodoOptim, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', learningRate, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
numIterationsPerEpoch = floor(DIM1 / miniBatchSize);

% cross-validation loop for each fold
for fold = 1:NF
    close all force
    
    % define the training and test sets for the current fold
    trainPattern = DIV(fold, 1:DIM1);
    testPattern = DIV(fold, DIM1+1:DIM2);
    y = yE(DIV(fold, 1:DIM1)); % training labels
    yy = yE(DIV(fold, DIM1+1:DIM2)); % test labels
    
    % convert categorical labels to numeric values
    y_numeric = double(y);
    yy_numeric = double(yy);
    
    % calculate the number of classes
    numClasses = numel(categories(y));
    
    % create the training set by resizing the images and stacking them
    clear nome trainingImages
    for pattern = 1:DIM1
        IM = NX{DIV(fold, pattern)}; % get the image for the current pattern
        IM = imresize(IM, [siz(1) siz(2)]); % resize images to match the input size of the network
        
        % if the image is grayscale, duplicate the channels to make it RGB
        if size(IM, 3) == 1
            IM(:,:,2) = IM;
            IM(:,:,3) = IM(:,:,1);
        end
        
        % Stack the images into a 4D array (for training)
        trainingImages(:,:,:,pattern) = IM;
    end
    imageSize = size(IM);

    % apply MixUp augmentation to the training set
    alpha = 0.2;
    [trainingImages, y_numeric] = mixupData(trainingImages, y_numeric, alpha);
    
    % apply CutMix augmentation to the training set
    beta = 0.2;
    [trainingImages, y_numeric] = cutmixData(trainingImages, y_numeric, beta);
    
    % round the numeric labels to the nearest integer
    y_numeric = round(y_numeric);
    
    % ensure the labels are within the valid range of classes
    y_numeric(y_numeric < 1) = 1;
    y_numeric(y_numeric > numClasses) = numClasses;
    
    % convert the numeric labels back to categorical labels
    y = categorical(y_numeric, 1:numClasses, categoryNames);

    % ensure that the number of images matches the number of labels
    assert(size(trainingImages, 4) == numel(y), 'Mismatch between number of images and labels');
    
    % apply standard image augmentation techniques
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandXScale', [1 2], ...
        'RandYReflection', true, ...
        'RandYScale', [1 2], ...
        'RandRotation', [-10 10], ...
        'RandXTranslation', [0 5], ...
        'RandYTranslation', [0 5]);
    
    % augment the training images using the image augmenter
    trainingImages = augmentedImageSource(imageSize, trainingImages, y, 'DataAugmentation', imageAugmenter);
    
    % fine-tune the pre-trained network by replacing the last few layers
    layersTransfer = net.Layers(1:end-3); % extract all layers except the last three
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];
    
    % train the network with the augmented training set
    netTransfer = trainNetwork(trainingImages, layers, options);
    
    % prepare the test set
    clear nome test testImages
    for pattern = ceil(DIM1)+1:ceil(DIM2)
        IM = NX{DIV(fold, pattern)}; % get the test image
        IM = imresize(IM, [siz(1) siz(2)]); % resize images to match the input size of the network
        
        % if the image is grayscale, duplicate the channels to make it RGB
        if size(IM, 3) == 1
            IM(:,:,2) = IM;
            IM(:,:,3) = IM(:,:,1);
        end
        
        % stack the test images into a 4D array
        testImages(:,:,:,pattern-ceil(DIM1)) = uint8(IM);
    end
    
    % classify the test patterns using the fine-tuned network
    [outclass, score{fold}] = classify(netTransfer, testImages);
    
    % calculate the accuracy for this fold
    [a, b] = max(score{fold}');
    ACC(fold) = sum(b == yy_numeric) / length(yy_numeric);
    
end

% calculate the final average accuracy across all folds
finalAccuracy = mean(ACC);
fprintf('Final Test Accuracy: %.4f\n', finalAccuracy);

% function to apply MixUp augmentation
function [mixedImages, mixedLabels] = mixupData(images, labels, alpha)
    % number of images in the dataset
    numImages = size(images, 4);

    % sample lambda from the beta distribution with parameter alpha
    lambda = betarnd(alpha, alpha, [numImages, 1]);

    % permute the images and labels randomly
    perm = randperm(numImages);

    % initialize the mixed images and labels
    mixedImages = images;
    mixedLabels = labels;

    % for each image, combine it with a randomly chosen image using MixUp
    for i = 1:numImages
        % blend the images using the sampled lambda value
        mixedImages(:,:,:,i) = lambda(i) * images(:,:,:,i) + (1 - lambda(i)) * images(:,:,:,perm(i));

        % blend the labels using the same lambda value
        mixedLabels(i) = lambda(i) * labels(i) + (1 - lambda(i)) * labels(perm(i));
    end
end

% function to apply CutMix augmentation
function [cutmixedImages, cutmixedLabels] = cutmixData(images, labels, beta)
    % number of images in the dataset
    numImages = size(images, 4);
    
    % sample lambda from the beta distribution with parameter beta
    lambda = betarnd(beta, beta, [numImages, 1]);

    % permute the images and labels randomly
    perm = randperm(numImages);

    % initialize the cutmixed images and labels
    cutmixedImages = images;
    cutmixedLabels = labels;

    % for each image, apply the CutMix augmentation
    for i = 1:numImages
        % calculate the size of the patch to cut based on lambda
        cutRatio = sqrt(1 - lambda(i));
        cutWidth = floor(cutRatio * size(images, 2));
        cutHeight = floor(cutRatio * size(images, 1));

        % randomly choose the position of the patch
        x = randi([1, size(images, 2) - cutWidth + 1]);
        y = randi([1, size(images, 1) - cutHeight + 1]);

        % replace the selected patch in the image with a patch from another image
        cutmixedImages(y:y+cutHeight-1, x:x+cutWidth-1, :, i) = ...
            images(y:y+cutHeight-1, x:x+cutWidth-1, :, perm(i));

        % combinee the labels based on the patch area ratio
        cutmixedLabels(i) = lambda(i) * labels(i) + (1 - lambda(i)) * labels(perm(i));
    end
end
