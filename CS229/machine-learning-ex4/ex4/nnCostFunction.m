function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Theta1 is [25X401]
% Theta2 is [10X26]
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feedforward implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input matrix, add the bias (1) column
A1 = [ones(m, 1) X];    % A1 is [5000x401]
Z2 = A1 * Theta1';      % Z2 is [5000x25]

% Hidden layer 
A2temp = sigmoid(Z2);               % A2temp is [5000x25]
A2 = [ones(size(Z2, 1), 1) A2temp]; % add the bias (1) column A2 is [5000x26]
Z3 = A2*Theta2';                    % Z2 is [5000x10]

% Output layer
H = sigmoid(Z3);                    % H is [5000x10]

% Recode the labels as vectors containing only values 0 or 1
Identity = eye(num_labels);    % Diagonal identity matrix [10x10]
Y = zeros(m, num_labels);      % Create an empty output matrix of [5000x10]

% Loop across all the dataset and replace with a new row [1x10]
% By using Identity matrix with a row pointer defined by the value of y(i)
% We make sure that we pick the row where the 1 will be in the right column
% For example, if y(23) = 6, then Identity(6,:) will have a 1 on column 6
for i = 1:m                         
  Y(i,:)= Identity(y(i),:);         
end

% Calculate the Cost fucntion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inner sum is across the K labels (we use 2 argument), J is [5000,1]
J =  sum((-Y).*log(H) - (1-Y).*log(1-H), 2);

% Outer sum is across m for the remaining vector
J = sum(J)/m;

% Add regularization, skip of the 1st terms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reg1 = Theta1(:, 2:end) .^ 2;  % [25x400]
Reg1 = sum(Reg1, 2);          % [25x1]
Reg1 = sum(Reg1);             % Scalar

Reg2 = Theta2(:, 2:end) .^ 2;  % [10x25]
Reg2 = sum(Reg2, 2);          % [10x1]
Reg2 = sum(Reg2);             % Scalar

Regularization = lambda * (Reg1 + Reg2) / (2*m);
J = J + Regularization;

% Output Layer Error
D3 = H - Y;

% Hidden Layer Error - Equation from page 8
temp1 = D3*Theta2;  % [5000x26] = [5000x10] * [10x26]
temp2 = sigmoidGradient([ones(size(Z2, 1), 1) Z2]);
D2 = (temp1 .* temp2);
D2 = D2(:, 2:end);      % Remove the bias column

Delta1 = D2'*A1;    % [25x401] = [25x5000] * [5000x401]
Delta2 = D3'*A2;    % [10x26] = [10x5000] * [5000x26]

% Unregularized
Theta1_grad = Delta1./m;    % [25x401]
Theta2_grad = Delta2./m;    % [10x26]

% Regularize put zeros in the 1st column and keep all other columns from
% Theta1 and Theta 2
gradientReg1 = (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
gradientReg2 = (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + gradientReg1;
Theta2_grad = Theta2_grad + gradientReg2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
