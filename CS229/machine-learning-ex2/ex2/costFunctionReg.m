function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


z = X * theta;   % z is a vector (m x 1)

% h(x) = g(theta'*X) where g is the sigmoid function
g = sigmoid(z);                 % g is a vector (m x 1)

% Calculate the cost fucntion
% temp1, temp2 are vectors (m x 1) and temp3 is a scalar
temp1 = (-y)  .* log(g);        % if y(i) = 1-> temp1 will be nonzero
temp2 = (1-y) .* log(1-g);      % if y(i) = 0-> temp2 will be nonzero
temp3 = ( theta' * theta  ) * lambda/(2*m) - ( theta(1) * theta(1) * lambda ) / (2*m);
J = sum( temp1 - temp2 ) / m + temp3;

% Gradient calculations
grad(1) =   (sum((g - y) .* X(:,1))) / m ;      % Should not regularize theta(1)

for j = 2:length(theta)                         % Loop from theta 2 to n
    temp4 = (lambda * theta(j)) / m;            % temp4 is a scalar
    grad(j) =   ( sum((g - y) .* X(:,j)) ) / m + temp4;
end


% =============================================================

end
