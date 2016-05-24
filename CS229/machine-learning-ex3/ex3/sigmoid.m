function g = sigmoid(z)
%   SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.
%   z can be a scalar, vector or matrix

g = 1.0 ./ (1.0 + exp(-z));
end
