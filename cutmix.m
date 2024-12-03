clear all
warning off

%scegli valore di datas in base a quale dataset vi serve
datas=29;

%carica dataset
load(strcat('DatasColor_',int2str(datas)),'DATA');
NF=size(DATA{3},1); %number of folds
DIV=DATA{3};%divisione fra training e test set
DIM1=DATA{4};%numero di training pattern
DIM2=DATA{5};%numero di pattern
yE=DATA{2};%label dei patterns
NX=DATA{1};%immagini

% Define category names based on classes
categoryNames = {'CLL', 'FL', 'MCL'};

% Convert numeric labels to categorical using defined category names
yE = categorical(yE, 1:3, categoryNames);

%carica rete pre-trained
net = alexnet;  %load AlexNet
siz=[227 227];
%se riesci con i tempi computazionali prova:
%net = vgg16;
%siz=[224 224];

%parametri rete neurale
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim='sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(DIM1/miniBatchSize);

for fold=1:NF
    close all force
    
    trainPattern=(DIV(fold,1:DIM1));
    testPattern=(DIV(fold,DIM1+1:DIM2));
    y=yE(DIV(fold,1:DIM1));%training label
    yy=yE(DIV(fold,DIM1+1:DIM2));%test label
    
    % Convert categorical labels to numeric
    y_numeric = double(y);
    yy_numeric = double(yy);
    
    % Calculate the number of classes
    numClasses = numel(categories(y));
    
    %creo il training set
    clear nome trainingImages
    for pattern=1:DIM1
        IM=NX{DIV(fold,pattern)};%singola data immagine
        
        IM=imresize(IM,[siz(1) siz(2)]);%si deve fare resize immagini per rendere compatibili con CNN
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        trainingImages(:,:,:,pattern)=IM;
    end
    imageSize=size(IM);
    
    % Apply MixUp augmentation
    alpha = 0.2;
    [trainingImages, y_numeric] = mixupData(trainingImages, y_numeric, alpha);
    
    % Apply CutMix augmentation
    beta = 0.2;
    [trainingImages, y_numeric] = cutmixData(trainingImages, y_numeric, beta);
    
    % Round numeric labels to the nearest integer
    y_numeric = round(y_numeric);
    
    % Ensure labels are within valid range
    y_numeric(y_numeric < 1) = 1;
    y_numeric(y_numeric > numClasses) = numClasses;
    
    % Convert augmented numeric labels back to categorical
    y = categorical(y_numeric, 1:numClasses, categoryNames);

    % Ensure the size of trainingImages and y are consistent
    assert(size(trainingImages, 4) == numel(y), 'Mismatch between number of images and labels');
    
    %creazione pattern aggiuntivi mediante tecnica standard
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[1 2], ...
        'RandYReflection',true, ...
        'RandYScale',[1 2],...
        'RandRotation',[-10 10],...
        'RandXTranslation',[0 5],...
        'RandYTranslation', [0 5]);
    trainingImages = augmentedImageSource(imageSize, trainingImages, y, 'DataAugmentation', imageAugmenter);
    
    %tuning della rete
    % The last three layers of the pretrained network net are configured for 1000 classes.
    %These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImages, layers, options);
    
    %creo test set
    clear nome test testImages
    for pattern=ceil(DIM1)+1:ceil(DIM2)
        IM=NX{DIV(fold,pattern)};%singola data immagine
        
        IM=imresize(IM,[siz(1) siz(2)]);
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        testImages(:,:,:,pattern-ceil(DIM1))=uint8(IM);
    end
    
    %classifico test patterns
    [outclass, score{fold}] =  classify(netTransfer, testImages);
    
    %calcolo accuracy
    [a, b] = max(score{fold}');
    ACC(fold) = sum(b==yy_numeric)./length(yy_numeric);
    
    %salvate quello che vi serve
    %%%%%
    
end
% Calculate the final average accuracy across all folds
finalAccuracy = mean(ACC);

% Display the final accuracy on the test set
fprintf('Final Test Accuracy: %.4f\n', finalAccuracy);
% got the Final Test Accuracy: 0.6107

% Function to apply MixUp augmentation
function [mixedImages, mixedLabels] = mixupData(images, labels, alpha)
    % Number of images
    numImages = size(images, 4);

    % Sample lambda from the beta distribution
    lambda = betarnd(alpha, alpha, [numImages, 1]);

    % Permutation of images and labels
    perm = randperm(numImages);

    % Initialize mixed images and labels
    mixedImages = images;
    mixedLabels = labels;

    for i = 1:numImages
        % Mix the images
        mixedImages(:,:,:,i) = lambda(i) * images(:,:,:,i) + (1 - lambda(i)) * images(:,:,:,perm(i));

        % Mix the labels
        mixedLabels(i) = lambda(i) * labels(i) + (1 - lambda(i)) * labels(perm(i));

    end
    
end

% Function to apply CutMix augmentation
function [cutmixedImages, cutmixedLabels] = cutmixData(images, labels, beta)
    % Number of images
    numImages = size(images, 4);
    
    % Sample lambda from the beta distribution
    lambda = betarnd(beta, beta, [numImages, 1]);

    % Permutation of images and labels
    perm = randperm(numImages);

    % Initialize cutmixed images and labels
    cutmixedImages = images;
    cutmixedLabels = labels;

    for i = 1:numImages
        % Calculate the size of the patch
        cutRatio = sqrt(1 - lambda(i));
        cutWidth = floor(cutRatio * size(images, 2));
        cutHeight = floor(cutRatio * size(images, 1));

        % Choose the position of the patch
        x = randi([1, size(images, 2) - cutWidth + 1]);
        y = randi([1, size(images, 1) - cutHeight + 1]);

        % Replace the patch in the image
        cutmixedImages(y:y+cutHeight-1, x:x+cutWidth-1, :, i) = ...
            images(y:y+cutHeight-1, x:x+cutWidth-1, :, perm(i));

        % Combine the labels
        cutmixedLabels(i) = lambda(i) * labels(i) + (1 - lambda(i)) * labels(perm(i));
    end
end
