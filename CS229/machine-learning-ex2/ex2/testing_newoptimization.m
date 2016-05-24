% Testing new optimization approach
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunctionHugo, initialTheta, options)

% exitFlag = 1 is converge
% functionVal is your cost value