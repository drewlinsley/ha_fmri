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

end

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

end