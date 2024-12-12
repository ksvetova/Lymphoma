% Load new dataset
load('GeneratedImagesMat.mat', 'GeneratedImages');

% Check the size of DATA to confirm the structure
NX = GeneratedImages{1}; % Images
yE = GeneratedImages{2}; % Labels

% Parameters for train/test split
splitRatio = 0.8;  % 80% for training, 20% for testing
numImages = length(NX);  % Total number of images
numTrain = round(splitRatio * numImages);  % Number of training images
numTest = numImages - numTrain;  % Number of test images

% Randomly shuffle the indices
rng(42);  % Set random seed for reproducibility
indices = randperm(numImages);
trainIndices = indices(1:numTrain);  % Training indices
testIndices = indices(numTrain+1:end);  % Test indices

% Create training and test sets
trainingImages = zeros([227, 227, 3, numTrain], 'like', NX{1});  % Preallocate the image matrix for training
testImages = zeros([227, 227, 3, numTest], 'like', NX{1});  % Preallocate the image matrix for testing

% Fill in the training set
for i = 1:numTrain
    IM = NX{trainIndices(i)};
    IM = imresize(IM, [227, 227]);  % Resize image to match AlexNet input size
    if size(IM, 3) == 1
        IM(:,:,2) = IM;
        IM(:,:,3) = IM(:,:,1);
    end
    trainingImages(:,:,:,i) = IM;
end

% Fill in the test set
for i = 1:numTest
    IM = NX{testIndices(i)};
    IM = imresize(IM, [227, 227]);
    if size(IM, 3) == 1
        IM(:,:,2) = IM;
        IM(:,:,3) = IM(:,:,1);
    end
    testImages(:,:,:,i) = IM;
end

% Create training labels and test labels
trainingLabels = categorical(yE(trainIndices));
testLabels = categorical(yE(testIndices));

% Network parameters (same as before)
net = alexnet;
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim = 'sgdm';
options = trainingOptions(metodoOptim, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', learningRate, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Fine-tune the network
layersTransfer = net.Layers(1:end-3);
numClasses = numel(unique(trainingLabels));  % Update the number of classes

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];
netTransfer = trainNetwork(trainingImages, trainingLabels, layers, options);

% Classify test images
[outclass, score] = classify(netTransfer, testImages);

% Calculate accuracy
[~, predictedLabels] = max(score, [], 2);
accuracy = sum(predictedLabels == double(testLabels)) / numTest;
disp(['Test accuracy: ', num2str(accuracy)]);
