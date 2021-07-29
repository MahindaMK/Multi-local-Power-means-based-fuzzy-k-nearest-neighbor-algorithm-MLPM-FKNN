function [predicted, memberships, numhits] = mlpm_fknn_updated(xtrain, ytrain, xtest, ytest, K, p)

% Updated multi-local power means-based fuzzy k-nearest neighbor (MLPM-FKNN) algorithm
% based on the study:

    % Kumbure, M. M., Lohrmann, C., Luukka, P.: A Study on Relevant Features for Intraday S&P 500 Prediction 
    % Using a Hybrid Feature Selection Approach. International Conference on Machine Learning, 
    % Optimization, and Data Science (LOD - 2021), Grasmere, Lake District, England – UK (2021). 

% INPUTS:
    % xtrain: Train data is a n-by-m data matrix consisting of n samples and m features(variables)
    % ytrain: n dimensional class vector of xtrain data (class labels should be in numerical form, eg. 1,2)
    % xtest: Test data is a D-by-m data matrix consisting of D samples and m features
    % ytest: D dimensional class vector of xtest data
    % K: Number of nearest neighbors to be selected
    % p: Parameter value for Power mean operator

% OUTPUTS:
    % predicted: Predicted class label for each test sample in xtest
    % memberships: Fuzzy class memberships values for each test sample in xtest
    % numhits: Number of correctly predicted test samples
    
    % 'pmean.m' is needed. 
    % This file is needed to compute power mean vectors of the set of nearest neighbor in each class


% Created by Mahinda Mailagaha Kumbure, 1/2021 

% Start

num_train = size(xtrain,1); % Find the number of samples in the train set
num_test  = size(xtest,1);  % Find the number of samples in the test set

m = 2.0; % scaling factor for fuzzy weights

max_class = max(ytrain);

% Aallocate space for storing predicted labels, resulted numhits and memberships
predicted = zeros(num_test, length(K));
numhits = zeros(length(K),1);
memberships = zeros(num_test, max_class, length(K));

% For each test samples, do:

for i = 1:num_test
    
    % Take the class labels
    class_labels = unique(ytrain);
    
    % Go through in each class in the training data
    for ii=1:length(class_labels)
    train_data_class_ii = xtrain(ytrain==class_labels(ii),:); % Find the train samples in class ii
    num_train_ii  = size(train_data_class_ii,1); % Take the number of samples in class ii
    
    % Computer the Euclidean distances from the test sample to each train pattern in class ii, 
    % (for efficiency, no need to take sqrt since it is a non-decreasing function)
    distances   = (repmat(xtest(i,:), num_train_ii,1) - train_data_class_ii).^2;
    distances = sum(distances,2)';

    [~, indeces] = sort(distances); % Sort the distances

    % Find the indexes of nearest neighbors from class ii
    if (num_train_ii<K) % if the # nearest neighbors defined is higher than # train samples in the class ii
    neighbor_index = indeces;    
    else
    neighbor_index = indeces(1:K);
    end
        
    % Computation of the locam power means for the nearest neighbors
    newdata = train_data_class_ii(neighbor_index, :);
    p_mean = pmean(newdata, p);
    local_mean(ii,:) = p_mean;
    lm_labels(ii)      =  class_labels(ii);
    end
    
    [n1,~] = size(local_mean); % Take the number of power mean vectors (n1)
    
    % Compute the Euclidean distances from the test sample to local-mean vectors found 
    % for each class
    distances = (repmat(xtest(i, :), n1, 1) - local_mean).^2;
    distances = sum(distances,2)';   
    
    % Compute fuzzy weights:
        % Though this weight calculation should be: 
        % weight = distances(neighbor_index).^(-2/(m-1)), 
        % but since we did not take sqrt above and the inverse 
        % 2th power the weights are: weight = sqrt(distances(neighbor_index)).^(-2/(m-1));
        % which is equaliavent to:
 	    weight = distances.^(-1/(m-1));
 
 	    % Set the Inf (infite) weights, if there are any, to  1.
 	    if max(isinf(weight))
            weight(isinf(weight)) = 1;
 	    end

    % Convert class ytrain to unary membership vectors (of 1s and 0s)
    labels_iter = zeros(length(lm_labels), max_class);
    for ii=1:n1
        labels_iter(ii,:) = [zeros(1, lm_labels(ii)-1) 1 zeros(1, max_class - lm_labels(ii))];
    end    
    
        test_out = weight*labels_iter/(sum(weight));
    
    clear lm_labels labels_iter tmp

    memberships(i,:,1) = test_out;  % Store memberships
    
    % Find the predicted class (the one with the max. fuzzy vote)
	[~, index_of_max] = max(test_out');
    predicted(i,1) = index_of_max;

    % Compute current hit rate, if test labels are given
        if ~isempty(ytest) && predicted(i,1)==ytest(i)
            numhits = numhits+1;
        end
    
end 
end
     