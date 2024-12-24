%clear all
%warning off


%carica dataset
load('GeneratedImagesMat.mat', 'GeneratedImages');


% Create the folders and decide the values for train and test set
GeneratedImages{4} = 780;
GeneratedImages{5} = 990;

numFolders = 5;

[m, n] = size(GeneratedImages{1});

%for i=1:numFolders
%    idx = randperm(n);
%    for k=1:length(idx)
%        GeneratedImages{1,3}{i,k} = idx(k);
%    end
%end

for i = 1:numFolders
    idx = randperm(n);  % Random permutation of integers 1 to n
    GeneratedImages{1,3}(i, :) = idx;  % Store the entire idx array directly in the matrix
end

NF=size(GeneratedImages{3},1); %number of folds
DIV=GeneratedImages{3};%divisione fra training e test set
DIM1=GeneratedImages{4};%numero di training pattern
DIM2=GeneratedImages{5};%numero di pattern
yE=GeneratedImages{2};%label dei patterns
NX=GeneratedImages{1};%immagini

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
    numClasses = max(y);%number of classes
    
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
    
    
    %inserire qui funzione per creare pose aggiuntive, in input si prende
    %(trainingImages,y) e in output restituisci una nuova versione di
    %(trainingImages,y) aggiornata con nuove immagini
    %%%%%%%%%%%


    initial_train_size = size(trainingImages, 4);

    % Image flipping
    for i=1:initial_train_size
        trainingImages(:, :, :, i) = fliplr(trainingImages(:, :, :, i));
    end

    % Image cropping
    for i=1:initial_train_size
        trainingImages(:, :, :, i) = imresize(imcrop(trainingImages(:, :, :, i), [50 50 227 227]), [227 227]);
    end



   % DO OTHERS
    
    %creazione pattern aggiuntivi mediante tecnica standard
    imageAugmenter = imageDataAugmenter();
    
    trainingImages = augmentedImageDatastore(imageSize,trainingImages,categorical(y'),'DataAugmentation',imageAugmenter);


    %tuning della rete


    % The last three layers of the pretrained network net are configured for 1000 classes.
    %These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImages,layers,options);
    
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
    [outclass, score{fold}] =  classify(netTransfer,testImages);
    
    %calcolo accuracy
    [a,b]=max(score{fold}');
    ACC(fold)=sum(b==yy)./length(yy);
    
    %salvate quello che vi serve
    %%%%%
    
end
