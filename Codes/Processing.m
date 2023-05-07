"""
This File contains the code used for processing the retinal fundus images.
Author: 
    Shams Nafisa Ali (snafisa.bme.buet@gmail.com)
Version: 07.05.2023
This code is licensed under the terms of the MIT-license.
"""

clc
clear all
close all

source_path = 'F:\OCT_Data\Raw_2class_All\'
destination_path = 'F:\OCT_Data\Processed_2class_All\'

% Name the folders inside the source and destination directories
folder = {'Healthy', 'Diseased'}; 

% Set the parameters
threshold = 0.01;
kernel_size = [51 51];
sigma = 2;
hsize = 21;


for i=1:length(folder)
    image_files = dir([source_path,folder{i},'\*.jpg']);
    nfiles = length(image_files)

    for j=1:nfiles

        currentfilename=image_files(j).name
        f = fullfile(source_path,folder{i},currentfilename);
        I = im2double(f);

        % Split the image into its red, green, and blue channels
        R = I(:,:,1);
        G = I(:,:,2);
        B = I(:,:,3);

        % Invert the green channel
        G_inv = 1 - G;
        zeroIndices = find(G < threshold);
        G_inv(zeroIndices) = G(zeroIndices);
        
        %% Illumination Equalization
        % Apply a mean filter to the green channel
        h = fspecial('average', kernel_size);
        bg = imfilter(G_inv, h, 'replicate');

        % Subtract the background image from the original image
        I_corr = G_inv - bg;

        % Add the mean intensity of the original image to the corrected image
        u = mean(G_inv(:));
        I_corr = I_corr + u;

        %% Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        I_clahe = adapthisteq(I_corr);

        %%  Smoothing/Denoising
        % Create the Gaussian filter
        G = fspecial('gaussian',[hsize hsize],sigma);

        % Apply the Gaussian filter to the image
        denoised_image = imfilter(I_clahe,G,'same');
  
        %% Normalization
        I_norm = double(denoised_image)/double(max(denoised_image(:)));
        figure; imshow(I_norm);
        f1 = fullfile(destination_path,folder{i}, strcat('Processed_',currentfilename));
        imwrite(I_norm,f1);
    end
end




"""
Inspired from the processing mentioned following article : 
Wu, Bo, et al. Automatic detection of microaneurysms in retinal fundus images.
Computerized Medical Imaging and Graphics 55 (2017): 106-112.
"""
