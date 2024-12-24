% WARNING: in this script we consider the 3 classes' generated images
% to be the SAME NUMBER for every label


% Create a folder to save individual generated images
addpath(pwd);
generatedImageFolder = 'generated_images';
if ~exist(generatedImageFolder, 'dir')
    mkdir(generatedImageFolder);  % Create the folder if it doesn't exist
end

num_images = 0;
label_index = 0;
for idxClass = 1:3
    numObservationsNew = 330;
    %idxClass = 1;
    ZNew = randn(numLatentInputs,numObservationsNew,"single");
    TNew = repmat(single(idxClass),[1 numObservationsNew]);
    
    ZNew = dlarray(ZNew,"CB");
    TNew = dlarray(TNew,"CB");
    
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        ZNew = gpuArray(ZNew);
        TNew = gpuArray(TNew);
    end
    
    XGeneratedNew = predict(netG,ZNew,TNew);

    
    % Set the new image size to 417x312 (you can adjust this if necessary)
    newImageSize = [312 417 3];  % For RGB images, this is [height, width, channels]
    
    
    % Loop through each generated image and save individually
    for imgIdx = 1:size(XGeneratedNew, 4)
        % Extract individual image
        img = squeeze(extractdata(XGeneratedNew(:, :, :, imgIdx)));
        
        % Rescale the image to [0, 1] range
        img = rescale(img);
    
        % Resize the image to the target size [417, 312]
        imgResized = imresize(img, newImageSize(1:2));  % Resize to [417, 312]
    
        % Let's keep track of the current image number we are considering
        current_num_image = (imgIdx + (idxClass - 1) * numObservationsNew);

        % Construct a unique filename for each generated image
        finalImagePath = fullfile(generatedImageFolder, ...
            sprintf('final_generated_img%d.png', current_num_image));
    
        % Save the individual image
        imwrite(imgResized, finalImagePath);
    end

    num_images = num_images + numObservationsNew;
    %label_index = label_index + numObservationsNew;

end


% Define the folder where the images are stored
imageFolder = '/MATLAB Drive/CGANs/generated_images';

% List all image files in the folder
imageFiles = dir(fullfile(imageFolder, '*.png'));

% Create a cell array with 5 cells (cell(1, 5))
GeneratedImages = cell(1, 5); 

% Initialize an empty cell array to store the images
images = cell(1, numel(imageFiles)); % Preallocate the cell array for images

label_indexes = zeros(1, num_images);
for current_image=1:num_images
    % Read each image
    img = imread(fullfile(imageFolder, imageFiles(current_image).name));
            
    % Store the image in the cell array
    images{current_image} = img;
    
   
    label_indexes(current_image) = floor(current_image/numObservationsNew) + 1;
end
label_indexes(num_images) = 3; %inconsistent, but useful in order to put the last label to 3 (it would be 4 instead)

%label_indexes


% Store all images in the first cell of myCellArray
GeneratedImages{1} = images;

% Store labels in the second cell of the mat file
GeneratedImages{2} = label_indexes;

%GeneratedImages{3} = DATA{3};
%GeneratedImages{4} = DATA{4};
%GeneratedImages{5} = DATA{5};

% Save the cell array to a .mat file
save('GeneratedImagesMat.mat', 'GeneratedImages');