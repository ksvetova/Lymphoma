datas = 29;
load(strcat('DatasColor_', int2str(datas)), 'DATA');

% Extracting images and labels
NX = DATA{1}; % Images
yE = DATA{2}; % Labels

% Number of images we want to display
numImages = 20;

% Get unique labels
uniqueLabels = unique(yE);

% Create a figure
figure;

% Initialize the counter for displayed images
imageCount = 0;

% Set the maximum number of images per label
maxImagesPerLabel = floor(numImages / length(uniqueLabels));

for labelIdx = 1:length(uniqueLabels)
    % Get the current label
    currentLabel = uniqueLabels(labelIdx);
    
    % Find indices of images with the current label
    labelIndices = find(yE == currentLabel);
    
    % Select the minimum between maxImagesPerLabel and the number of available images
    numAvailableImages = min(maxImagesPerLabel, length(labelIndices));
    
    for i = 1:numAvailableImages
        imageCount = imageCount + 1;
        subplot(4, 5, imageCount); % Create a 4x5 grid for 20 images
        imshow(NX{labelIndices(i)}); % Display the image
        title(['Label: ', num2str(currentLabel)]); % Add label to the title
        
        % If we have reached the maximum number of images, exit the loop
        if imageCount >= numImages
            break;
        end
    end
    
    % If we have reached the maximum number of images, exit the loop
    if imageCount >= numImages
        break;
    end
end
