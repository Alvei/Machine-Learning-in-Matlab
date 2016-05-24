function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%   regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%   theta is (n+1) x 1

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% h(x) = g(theta'*X) where g is the sigmoid function
% X is m x (n+1)
% theta is (n+1) x 1
z = X * theta;      % z is a vector (m x 1)
g = sigmoid(z);     % g is a vector (m x 1)

% Calculate the Cost fucntion
% temp1, temp2 are vectors (m x 1) and temp3 is a scalar
temp1 = (-y)  .* log(g);        % if y(i) = 1-> temp1 will be nonzero
temp2 = (1-y) .* log(1-g);      % if y(i) = 0-> temp2 will be nonzero
regterm = ( theta' * theta  ) * lambda/(2*m) - ( theta(1) * theta(1) * lambda ) / (2*m);
J = sum( temp1 - temp2 ) / m + regterm;

% Gradient calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vectorized implementation
temp4 = ones(n);
temp4(1) = 0;   % do not regularize theta_0 - theta(1) in Mathlab
grad = 1./m * X' * (g - y) + lambda * (theta .* temp4)/ m;

% older implementation using for loop

% grad(1) =   (sum((g - y) .* X(:,1))) / m ;      % Should not regularize theta(1)

% for j = 2:length(theta)                         % Loop from theta 2 to n
%    temp4 = (lambda * theta(j)) / m;             % temp4 is a scalar
%    grad(j) =   ( sum((g - y) .* X(:,j)) ) / m + temp4;
% end

% =============================================================

grad = grad(:);  % It was there, takes all the columns and adds them to create long vector

end
