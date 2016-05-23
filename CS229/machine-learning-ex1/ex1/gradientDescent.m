function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);                      % number of training examples
J_history = zeros(num_iters, 1);

n = length(theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % theta is n x 1
    % X is m x n
    % X * temp is m x 1
    % y is m x 1
    % X(:,i) is m x 1
    temp = theta;
    for i = 1: n
        delta  = (1/m) * sum( (X*temp - y ).*X(:,i) );
        theta(i) = temp(i) - alpha * delta;
    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
