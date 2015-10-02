function [stats,models] = train_model(input,X,folds,lambda)

if ~exist('folds','var'), folds = 10; end
if ~exist('lambda','var'), lambda = 10; end
if ~exist('wm','var'),wm = 1; end
input_inds = reshape(repmat(1:folds,ceil(numel(X(:,1))/folds),1),[],1);
if numel(input_inds) > numel(input{1}(1,1,:)),
    input_inds = input_inds(1:numel(input{1}(1,1,:)));
elseif numel(input_inds) < numel(input{1}(1,1,:)),
    input_inds = cat(1,input_inds,repmat(input_inds(end),numel(input{1}(1,1,:)) - numel(input_inds),1));
end

sp_lambda = speye(size(X,2) + 1,size(X,2) + 1) .* lambda;
sp_lambda(1) = 0;
ns = size(input{1});
models = cell(numel(input),1);
stats = models;
for idx = 1:folds,
    fprintf('Working fold %i/%i\r',idx,folds);
    training_ids = setdiff(1:folds,idx);
    testing_inds = ismember(input_inds,idx);
    training_inds = ismember(input_inds,training_ids);
    testing_data = cellfun(@(x) x(:,:,testing_inds),input,'UniformOutput',false);
    training_data = cellfun(@(x) x(:,:,training_inds),input,'UniformOutput',false);
    if wm == 1,
        %%Mean method
        train_X = X(training_inds,:);
        train_X = cat(2,ones(size(train_X,1),1),train_X);
        train_Y = mean(cat(4,training_data{:}),4);
        test_Y = mean(cat(4,testing_data{:}),4);
        test_X = X(testing_inds,:);
        test_X = cat(2,ones(size(test_X,1),1),test_X);
    elseif wm == 2,
        %%Cat method
        train_X = repmat(X(training_inds,:),numel(training_data),1);
        train_X = cat(2,ones(size(train_X,1),1),train_X);
        test_Y = cat(3,testing_data{:});
        train_Y = cat(3,training_data{:});
        test_X = repmat(X(testing_inds,:),numel(testing_data),1);
        test_X = cat(2,ones(size(test_X,1),1),test_X);
        %
    end
    it_models = zeros(ns(1),ns(2),size(train_X,2),'single');
    it_corr = zeros(ns(1),ns(2),'single');
    for x = 1:ns(1),
        for y = 1:ns(2),
            it_ytest = squeeze(test_Y(x,y,:));
            if ~isnan(sum(it_ytest)),
                it_ytrain = squeeze(train_Y(x,y,:));
                [it_ytrain,mu,sd] = zscore(it_ytrain);
                it_ytest = bsxfun(@rdivide,bsxfun(@minus,squeeze(test_Y(x,y,:)),mu),sd);
                it_m = (train_X'*train_X + sp_lambda) \ train_X' * it_ytrain;
                %it_m = lasso(train_X,it_ytrain,'Lambda',.01);
                it_models(x,y,:) = it_m;
                it_yhat = test_X * it_m;
                it_corr(x,y) = corr(it_ytest,it_yhat);
            end
        end
    end
    models{idx} = it_models;
    stats{idx} = it_corr;
end


return
%LASSO code
% load hald
% A = [ones(13,1),ingredients];
% fmincon(@(w) myobj(A,heat,w,1,0),rand(5,1),[],[],[],[],zeros(5,1))
%
% function out = myobj(A,a,w,beta,lambda)
% resid = a - A*w;
% out = 0.5*norm(resid)^2 + (beta/2)*norm(w)^2 + lambda*sum(abs(w));

