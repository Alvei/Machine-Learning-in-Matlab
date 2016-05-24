function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

incr = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]; % Increment suggested
incr_C = incr;
incr_Sig = incr;

best = zeros(3,1);  % Initialize vector that contains [redError, tryC, tryS]
trial = 1;          % Counter

% Iterate over all the increments of C
for tryC = incr_C
    % Iterate over all the increments of Sigma
	for tryS = incr_Sig
		fprintf(['Attempt #%d: C = %f, sigma = %f\n'], trial, tryC, tryS);
		
        % Run the SVM trainer using the Gaussian Kernel
		model = svmTrain(X, y, tryC, @(x1, x2) gaussianKernel(x1, x2, tryS));
        
        % Check prediction and prediction error
		prediction = svmPredict(model, Xval);  % Prediction is a vector[200x1]
		predError = mean(double(prediction == yval)); % if equal then 1 otherwise 0
        fprintf(['Attempt #%d: predError %f\n'],trial, predError)
        
        % if the new prediction error better than old save it
		if (predError > best(1))
			fprintf(['Best so far %f\n'],  predError);
			best = [predError, tryC, tryS];
        end
        
        trial = trial + 1;
	end
end

% Done looping
C = best(2);
sigma = best(3);
fprintf(['\n Best Found: C = %f, sigma = %f\n'], C, sigma);






% =========================================================================

end
