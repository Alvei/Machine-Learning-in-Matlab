function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Implementing J = sum of [(H(xi) - yi)^2] divided by 1/2m
% y is m x 1 vector
% h = X * theta is also a m x 1 vector
% X is m x n matrix
% theta is n X 1 (parameter matrix for a linear regression
h = X * theta;
J = sum(( h - y ) .^2 )/( 2 * m );

% Implementating regularization but removin the first term
regterm = ( theta' * theta  ) * lambda/(2*m) - ( theta(1) * theta(1) * lambda ) / (2*m);
J = J + regterm;

% Gradient calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(theta);
temp = ones(n);
temp(1) = 0;   % do not regularize theta_0 - theta(1) in Mathlab
grad = 1./m * X' * (h - y) + lambda * (theta .* temp)/ m;

% =========================================================================

grad = grad(:); % Takes all the columns and adds them to create long vector

end
