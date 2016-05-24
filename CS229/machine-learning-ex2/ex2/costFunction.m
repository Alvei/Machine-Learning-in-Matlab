function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X * theta;
g = sigmoid(z);  % h(x) = g(theta'*X) where g is the sigmoid function
temp1 = (-y)  .* log(g);        % if y(i) = 1-> temp1 will be nonzero
temp2 = (1-y) .* log(1-g);      % if y(i) = 0-> temp2 will be nonzero
J = sum( temp1 - temp2 ) / m;

% Gradient calculations
for j = 1:length(theta)
    grad(j) =   ( sum(( g - y ) .* X( :,j )) ) / m ;
end



% =============================================================

end
