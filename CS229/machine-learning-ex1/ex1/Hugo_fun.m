fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
theta = zeros(2, 1); % initialize fitting parameters

[theta, Jhistory] = gradientDescent(X, y, theta, alpha, iterations);
% print theta to screen

fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit

plotData(X, y);

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

plot(Jhistory)
legend('Cost function vs. iteration')

