function [predicted, memberships, numhits] = mlpm_fknn(xtrain, ytrain, xtest, ytest, K, p)

% Muli-local Power means-based fuzzy k-nearest neighbor (MLPM-FKNN) algorithm

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

% Reference:
    % Kumbure, M. M., Luukka, P., Collan, M.: An enhancement of fuzzy k-nearest neighbor classifier 
    % using multi-local power means. In: Proceeding of the 11th Conference of the European Society 
    % for Fuzzy Logic and Technology (EUSFLAT), pp. 83–	90, Atlantis Press (2019) 
    % https://doi.org/10.2991/eusflat-19.2019.13

% Created by Mahinda Mailagaha Kumbure, 1/2019 

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
    % Computer the Euclidean distances from test sample to each train
    % pattern, (for efficiency, no need to take sqrt since it is a
    % non-decreasing function)
    
    distances = (repmat(xtest(i,:), num_train,1) - xtrain).^2;
    distances = sum(distances,2)';

    [~, indeces] = sort(distances); % Sort the distances
    neighbor_index = indeces(1:K);  % Find the indexes of nearest neighbors
	weight = ones(1,length(neighbor_index));
    
    % Local mean computation for each class in the set of nearest neighbors
    newdata = xtrain(neighbor_index,:);
    [~,t1]  = size(newdata);
    nn_labels = ytrain(neighbor_index);
    
    class_index = unique(nn_labels);
    train_class_label = zeros(length(class_index),1);
    local_mean = zeros(length(class_index),t1);
   
    for c = 1:length(class_index)  % Go through each class
        class_train_sample = newdata(nn_labels == class_index(c), :); % Take train sample from class c
        local_mean(c,:)  = pmean(class_train_sample,p);
        train_class_label(c,1) = class_index(c);
    end
   
    lm_labels = train_class_label; % Correspond class label for each local mean vector
    [n1,~]    = size(local_mean);  % Take the number of power mean vectors (n1)
   
    % Compute the Euclidean distances from test pattern to local-mean (Bonferroni) vectors found 
    % for each class
    distances = (repmat(xtest(i,:), n1,1) - local_mean).^2;
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
    
    for ii = 1:n1
        labels_iter(ii,:) = [zeros(1, lm_labels(ii)-1) 1 zeros(1, max_class - lm_labels(ii))];
    end    
    
	test_out = weight*labels_iter/(sum(weight));
    
    clear lm_labels labels_iter tmp 

    memberships(i,:,1) = test_out; % store memberships
     
    
    % Find the predicted class (the one with the max. fuzzy vote)
    [~, index_of_max]  = max(test_out');
    predicted(i,1) = index_of_max;

    % Compute current hit rate, if test labels are given
    if ~isempty(ytest) && predicted(i,1)==ytest(i)
            numhits = numhits + 1;
    end
   
      
end
