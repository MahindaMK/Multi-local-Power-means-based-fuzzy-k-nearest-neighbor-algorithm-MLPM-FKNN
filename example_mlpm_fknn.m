% An example for the use of MLPM-FKNN classifier

clear all; close all; clc

% Load the data (example data of ionosphere)
load ionosphere
    % X: features
    % Y: cell array of the class labels (g:good and b:bad)

% Convert class labels to numeric 
Y      = categorical(Y);
labels = zeros(length(Y),1);
labels(Y=='g') = 1;
labels(Y=='b') = 2;

% If the input data contains negative values, then it is possible to get multi-local mean vectors 
% with complex values, for example, when p=1.5. 
% To avoid this issue, the data matrix needs to be normalized into 0 and 1 range. 

X = normalize(X,'range');

data = [X labels];


% Cross validation
val = 0.2; % Percentage for holdout validation
cv  = cvpartition(size(data,1),'HoldOut', val);
idx = cv.test;

% Separate to training and test data
Xtrain  = data(~idx,1:end-1); % train data with n patterns and m features
Ytrain  = data(~idx,end); % class labels of train patters 

Xtest   = data(idx,1:end-1); % test data with D patterns and m features
Ytest   = data(idx,end); % class labels of test patterns

K = 10;  % Initialization of the number of nearest neighbors
p = 3;   % Parameter p for Bonferroni mean operator
m = 1.5; % Fuzzy strength values

% MLPM-FKNN function call
[predicted, memberships, numhits] = mlpm_fknn(Xtrain, Ytrain, Xtest, Ytest, K, p);

% Classification accuracy
classification_accuracy  = numhits/length(Xtest)



