function [ C, L, U ] = SpectralClustering( input, k,Type )
%SPECTRALCLUSTERING Executes spectral clustering algorithm
%   Executes the spectral clustering algorithm defined by
%   Type on the adjacency matrix W and returns the k cluster
%   indicator vectors as columns in C.
%   If L and U are also called, the (normalized) Laplacian and
%   eigenvectors will also be returned.
%
%   'W' - Adjacency matrix, needs to be square
%   'k' - Number of clusters to look for
%   'Type' - Defines the type of spectral clustering algorithm
%            that should be used. Choices are:
%      1 - Unnormalized
%      2 - Normalized according to Shi and Malik (2000)
%      3 - Normalized according to Jordan and Weiss (2002)
%
%   References:
%   - Ulrike von Luxburg, "A Tutorial on Spectral Clustering", 
%     Statistics and Computing 17 (4), 2007
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

% calculate degree matrix
W = SimGraph_Full(input, .5);
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;

% compute normalized Laplacian if needed
switch Type
    case 2
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate inverse of D
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L;
    case 3
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate D^(-1/2)
        D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L * D;
end

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
diff   = eps;
[U, ~] = eigs(L, k, diff);
[U,S] = eig(L);

% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
if Type == 3
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
end

% now use the k-means algorithm to cluster U row-wise
% C will be a n-by-1 matrix containing the cluster number for
% each data point
% U
% size(U)
C = kmeans(U, k, 'start', 'cluster', ...
                 'EmptyAction', 'singleton', 'Display', 'off', 'MaxIter', 200);
             
% now convert C to a n-by-k matrix containing the k indicator
% vectors as columns
C = sparse(1:size(D, 1), C, 1);


function W = SimGraph_Full(M, sigma)
% SIMGRAPH_FULL Returns full similarity graph
%   Returns adjacency matrix for a full similarity graph where
%   a Gaussian similarity function with parameter sigma is
%   applied.
%
%   'M' - A d-by-n matrix containing n d-dimensional data points
%   'sigma' - Parameter for Gaussian similarity function
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

% Compute distance matrix
W = squareform(pdist(M'));

% Apply Gaussian similarity function
W = simGaussian(W, sigma);


function [ M ] = simGaussian( M, sigma )
%SIMGAUSSIAN Calculates Gaussian similarity on matrix
%   simGaussian(M, sigma) returns a matrix of the same size as
%   the distance matrix M, which contains similarity values
%   that are computed by using a Gaussian similarity function
%   with parameter sigma.
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

M = exp(-M.^2 ./ (2*sigma^2));

