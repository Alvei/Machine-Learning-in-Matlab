function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Implementing J = sum of [(H(xi) - yi)^2] divided by 1/2m
% y is m x 1 vector
% H = X * theta is also a m x 1 vector
% X is m x 2 matrix
% theta is 2 X 1 (parameter matrix for a linear regression

J = sum(( X * theta - y ) .^2 )/( 2 * m );


% =========================================================================

end
