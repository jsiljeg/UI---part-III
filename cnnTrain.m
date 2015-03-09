%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
% addpath ../common/;
images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================


%%======================================================================
%% STEP 2: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options);
%%======================================================================
%% STEP 3: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = (sum(preds==testLabels)/length(preds))*100;

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f percent\n',acc);