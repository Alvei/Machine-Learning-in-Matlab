function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
% In the example it is 10 classes (or possible "digit") => use Theta2 for
% output size
m = size(X, 1);
num_labels = size(Theta2, 1);    

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add ones or bias unit to the X data matrix
% Input layer has m unit = 5000 & 400+1 pixel => (5000 x 401)
a1 = [ones(m, 1) X];  

Z2 = Theta1*a1';  % Matrix Z2 = (25 x 401) * (401 x 5000) = (25 x 5000)
a2 = sigmoid(Z2);

% Adding bias unit as the 1st row to make a2 (26 x 5000)
m2 = size(a2,2);
a2 = [ones(1, m2); a2]; 

% Output layer
Z3 = Theta2*a2; % Z3 = (10 x 26) * (26 x 5000) = (10 x 5000)
a3 = sigmoid(Z3);


% max function returns the highest value of each row into a column vector
% for each learning set search the columns for the largest number and
% return x for the value and p for location, we use p b/c it tells us the
% digit. Eg. is for a row, the largest number is in column 3, then the
% digit is "3"
[x, p] = max(a3', [], 2);   % x and p are (5000x1)



% =========================================================================


end
