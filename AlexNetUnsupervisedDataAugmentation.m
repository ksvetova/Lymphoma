clear all
warning off

% Choose dataset
datas=29;

% Load dataset
load(strcat('DatasColor_', int2str(datas)), 'DATA');
NF = size(DATA{3}, 1); % number of folds
DIV = DATA{3}; % division between training and test tests
DIM1 = DATA{4}; % number of training patterns
DIM2 = DATA{5}; % number of total patterns
yE = DATA{2}; % labels of patterns
NX = DATA{1}; % images

% Load pre-trained network
% net = alexnet;  % load AlexNet
% siz=[227, 227];
%se riesci con i tempi computazionali prova:
net = vgg16;
siz=[224 224];

% Neural network parameters
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim = 'sgdm';
options = trainingOptions(metodoOptim, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', learningRate, ...
    'Verbose', false, ...
    'Plots','training-progress');
numIterationsPerEpoch = floor(DIM1/miniBatchSize);

for fold = 1:NF
    close all force
    
    trainPattern = (DIV(fold, 1:DIM1));
    testPattern = (DIV(fold, DIM1+1:DIM2));
    y = yE(DIV(fold, 1:DIM1)); % training label
    yy = yE(DIV(fold, DIM1+1:DIM2)); % test label
    numClasses = max(y); % number of classes
    
    % Create the training set
    clear nome trainingImages augmentedTrainingImages
    for pattern = 1:DIM1
        IM = NX{DIV(fold, pattern)}; % single image
        
        IM = imresize(IM, [siz(1), siz(2)]); % Resize image
        if size(IM, 3) == 1
            IM(:, :, 2) = IM;
            IM(:, :, 3) = IM(:, :, 1);
        end
        trainingImages(:, :, :, pattern) = IM; % Original images
    end
    imageSize = size(IM);
    
    %inserire qui funzione per creare pose aggiuntive, in input si prende
    %(trainingImages,y) e in output restituisci una nuova versione di
    %(trainingImages,y) aggiornata con nuove immagini
    %%%%%%%%%%%

    % Augmentation functions
    augmentations = {@flipImage, @rotateImage, @cropImage, @shiftImage, @colorJitter, @addNoise, @pcaJitter};

    % Apply augmentation
    augmentedTrainingImages = augmentDataset(trainingImages, augmentations);
    
    % Convert augmented data to categorical labels
    trainingImagesAugmented = augmentedImageDatastore(siz, augmentedTrainingImages, categorical(y'));

    % Transfer learning step (tuning of the network)
    % The last three layers of the pretrained network net are configured for 1000 classes.
    % These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImagesAugmented, layers, options);
    
    % Create test set
    clear nome test testImages
    for pattern = ceil(DIM1) + 1:ceil(DIM2)
        IM = NX{DIV(fold, pattern)}; % single image
        
        IM = imresize(IM, [siz(1), siz(2)]);
        if size(IM, 3) == 1
            IM(:, :, 2) = IM;
            IM(:, :, 3) = IM(:, :, 1);
        end
        testImages(:, :, :, pattern - ceil(DIM1)) = uint8(IM);
    end
    
    % classify test patterns
    [outclass, score{fold}] = classify(netTransfer, testImages);
    
    % Calculate accuracy
    [a, b] = max(score{fold}');
    ACC(fold) = sum(b == yy) ./ length(yy);
    
    % Save results if needed
    %%%%%    
end

% Augmentation Functions
function augmentedImages = augmentDataset(images, augmentations)
    % Apply augmentations to the dataset
    numImages = size(images, 4);
    augmentedImages = [];
    for i = 1:numImages
        img = images(:, :, :, i);
        numAugmentations = randi([1, length(augmentations)]); % Random number of augmentations
        selectedAugmentations = randsample(augmentations, numAugmentations);

        for j = 1:length(selectedAugmentations)
            img = selectedAugmentations{j}(img);
        end
    augmentedImages = cat(4, augmentedImages, img);
    end
end

function img = flipImage(img)
    img = fliplr(img);
end

function img = rotateImage(img)
    angle = randi([-90, 90]); % Random rotation
    img = imrotate(img, angle, 'bilinear', 'crop');
end

function img = cropImage(img)
    [h, w, ~] = size(img);
    cropSize = [round(0.8 * h), round(0.8 *w)];
    x = randi([1, w - cropSize(2)]);
    y = randi([1, h - cropSize(1)]);
    img = imresize(img(y:y+cropSize(1)-1, x:x+cropSize(2)-1, :), [h, w]);
end

function img = shiftImage(img)
    shiftX = randi([-10, 10]); % Random horizontal shift
    shiftY = randi([-10, 10]); % Random vertical shift
    tform = affine2d([1 0 0; 0 1 0; shiftX shiftY 1]);
    img = imwarp(img, tform, 'OutputView', imref2d(size(img)));
end

function img = colorJitter(img)
% Apply random hue, saturation, brightness, and contrast adjustments
    img = jitterColorHSV(img, ...
        'Hue', [-0.1, 0.1], ... % Adjust hue by 10%
        'Brightness', [0.8, 1], ... % Scale brightness by 0.8 to 1
        'Contrast', [0.9, 1.1], ... % Scale contrast by 0.9 to 1.1
        'Saturation', [0.8, 1]); % Scale brightness by 0.8 to 1
end

function img = addNoise(img)
    % Add Gaussian noise to the image
    noise = randn(size(img)) * 0.05; % Generate Gaussian noise (mean = 0, std = 0.05)
    img = double(img) + noise; % Convert image to double and add noise
    img = min(max(img, 0), 1); % Clip values to [0, 1]
    img = im2uint8(img); % Convert back to uint8
end

function img = pcaJitter(img)
    imgDouble = double(reshape(img, [], 3));
    [coeff, score, ~] = pca(imgDouble);
    noise = randn(size(score)) * 0.1;
    imgJittered = score + noise * coeff';
    img = reshape(imgJittered, size(img));
    img = uint8(min(max(img, 0), 255)); % Clip values
end

function image = method1dct(im)
    for i = 1:3
        % apply DCT to every dimension of the image
        DCT = dct2(im(:,:,i));
        d = DCT;
        % set some pixel to 0
        DCT(randi([0 1], size(DCT,1), size(DCT,2)) == 0) = 0;
        % leave unmodified pixel in position (1,1)
        DCT(1,1) = d(1,1);
        % apply inverse DCT
        image(:,:,i) = idct2(DCT);
    end
    image = uint8(image);
end 

function image = method2dct(im)
    [y, x, z] = size(im);

    % apply standard deviation for every level
    sigma2 = std(double(im));
    sigma1 = std(sigma2);
    sigma = std(sigma1) / 2;
    for channel = 1:z
        % apply DCT
        DCTim = dct2(im(:,:,channel));
        for riga = 1:y
            for colonna = 1:x
                % DTC(1,1) unmodified
                if ~(riga == 1 && colonna == 1)
                    % calculate random number between (-0.5,0.5)
                    random_z = rand();
                    while random_z > 1/2
                        random_z = rand();
                    end
                    prob = rand();
                    if prob > 1/2
                        random_z = random_z * 1;
                    else
                        random_z = random_z * -1;
                    end
                    %modify DCT image
                    DCTim(riga, colonna) = DCTim(riga, colonna) + sigma * random_z;
                end
            end
        end
        %inverse DCT
        image(:,:,channel) = idct2(DCTim);
    end
    image = uint8(image);
end


