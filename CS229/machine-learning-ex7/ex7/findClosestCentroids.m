function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%   X is [300 x 2]
%   centroids is [3 x 2]

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Iterate over all the row of matrix X
for i = 1:length(X)
	differences = zeros(K,1);   % Initialize the vector of difference
	x = X(i,:);                 % Take the data set of row i
    
	% Iterate over all the centroids
    for j = 1:K
		k = centroids(j,:);  % Take the location of the centroid j
		diff = x - k;
		differences(j) = diff * diff';
    end
    % Keep only the closest centroid and save in index idx(i)
	[y, idx(i)] = min(differences);  % y is discarded
	
end





% =============================================================

end

