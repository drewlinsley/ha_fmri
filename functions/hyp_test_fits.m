function p_vals = hyp_test_fits(masks,input,X,num_toss,lambda,folds)

if ~exist('folds','var'), folds = 10; end
if ~exist('lambda','var'), lambda = 10; end
if ~exist('lambda_folds','var'),lambda_folds = 10; end
if ~exist('wm','var'),wm = 1; end
if ~exist('num_toss','var'),num_toss = 5; end
if ~exist('num_iter','var'),num_iter = 200; end

input_inds = reshape(repmat(1:folds,ceil(numel(X(:,1))/folds),1),[],1);
if numel(input_inds) > numel(input{1}(1,1,:)),
    input_inds = input_inds(1:numel(input{1}(1,1,:)));
elseif numel(input_inds) < numel(input{1}(1,1,:)),
    input_inds = cat(1,input_inds,repmat(input_inds(end),numel(input{1}(1,1,:)) - numel(input_inds),1));
end

ns = size(input{1});
num_l = numel(lambda);
models = cell(numel(input),1);
stats = models;
train_stats = models;
train_lambda = models;
p_vals = models;
count = 1;
num_mask_vox = sum(sum(masks == 1));
warning off
%for idx = 1:folds,
for idx = folds:folds,
    fprintf('Working fold %i/%i\r',idx,folds);
    %Take each run, train on 80% and test on 20% (tossing the first 5%)
    training_ids = setdiff(1:folds,idx); %figure out training IDs
    testing_inds = ismember(input_inds,idx);
    training_inds = ismember(input_inds,training_ids);
    testing_inds_ids = find(testing_inds);
    testing_inds(testing_inds_ids([1:num_toss,numel(testing_inds_ids) - (num_toss-1):numel(testing_inds_ids)])) = 0;%remove the first 5 and last 5 testing to ensure there's no bleedover between runs
    testing_inds = testing_inds == 1;
    
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
    it_corr_train = it_corr;
    store_lambda = it_corr;
    it_pvals = ones(size(it_corr));
    for x = 1:ns(1),
        for y = 1:ns(2),
            if masks(x,y) == 1,
                %Get true score
                fprintf('Working voxel %i/%i\r',count,num_mask_vox)
                it_ytrain = squeeze(train_Y(x,y,:));
                [it_ytrain,mu,sd] = zscore(it_ytrain);
                it_ytest = bsxfun(@rdivide,bsxfun(@minus,squeeze(test_Y(x,y,:)),mu),sd);
                c = cvpartition(it_ytrain,'KFold',10);
                chosen_lambda = zeros(lambda_folds,1);
                for il = 1:lambda_folds,
                    ftrain = it_ytrain(c.training(il));
                    ftest = it_ytrain(c.test(il));
                    fX = train_X(c.training(il),:);
                    fXt = train_X(c.test(il),:);
                    f_itr = zeros(num_l,1);
                    for iil = 1:num_l,
                        flambda = speye(size(fX,2)) .* lambda(iil);
                        flambda(1) = 0;
                        f_itm = (fX'*fX + flambda) \ fX' * ftrain;
                        f_itr(iil) = corr(ftest,fXt*f_itm);
                    end
                    [~,chosen_lambda(il)] = max(f_itr);
                end
                chosen_lambda = mode(chosen_lambda);
                sp_lambda = speye(size(train_X,2)) .* lambda(chosen_lambda);
                sp_lambda(1) = 0;
                it_m = (train_X'*train_X + sp_lambda) \ train_X' * it_ytrain;
                %it_m = lasso(train_X,it_ytrain,'Lambda',.01);
                it_models(x,y,:) = it_m;
                it_yhat = test_X * it_m;
                it_corr(x,y) = corr(it_ytest,it_yhat);
                it_corr_train(x,y) = corr(it_ytrain,train_X * it_m);
                store_lambda(x,y) = lambda(chosen_lambda);
                perm_it_corr = get_null(train_Y,test_Y,train_X,test_X,lambda,lambda_folds,num_l,num_iter);
                it_pvals(x,y) = ((sum(it_corr(x,y) < perm_it_corr) + 1) ./ (num_iter + 1));
                count = count + 1;
            end
        end
    end
    models{idx} = it_models;
    stats{idx} = it_corr;
    train_stats{idx} = it_corr_train;
    train_lambda{idx} = store_lambda;
    p_vals{idx} = it_pvals;
end
warning on


function perm_it_corr = get_null(train_Y,test_Y,train_X,test_X,lambda,lambda_folds,num_l,num_iter)

perm_it_corr = zeros(num_iter,1);
for pi = 1:num_iter,
    it_ytrain = randn(numel(train_Y(1,1,:)),1); %since things are z-scored, we're matching 1st order statistics
    it_ytest = randn(numel(test_Y(1,1,:)),1);
    c = cvpartition(it_ytrain,'KFold',10);
    chosen_lambda = zeros(lambda_folds,1);
    for il = 1:lambda_folds,
        ftrain = it_ytrain(c.training(il));
        ftest = it_ytrain(c.test(il));
        fX = train_X(c.training(il),:);
        fXt = train_X(c.test(il),:);
        f_itr = zeros(num_l,1);
        for iil = 1:num_l,
            flambda = speye(size(fX,2)) .* lambda(iil);
            flambda(1) = 0;
            f_itm = (fX'*fX + flambda) \ fX' * ftrain;
            f_itr(iil) = corr(ftest,fXt*f_itm);
        end
        [~,chosen_lambda(il)] = max(f_itr);
    end
    chosen_lambda = mode(chosen_lambda);
    sp_lambda = speye(size(train_X,2)) .* lambda(chosen_lambda);
    sp_lambda(1) = 0;
    it_m = (train_X'*train_X + sp_lambda) \ train_X' * it_ytrain;
    %it_m = lasso(train_X,it_ytrain,'Lambda',.01);
    it_yhat = test_X * it_m;
    perm_it_corr(pi) = corr(it_ytest,it_yhat);
end



%LASSO code
% load hald
% A = [ones(13,1),ingredients];
% fmincon(@(w) myobj(A,heat,w,1,0),rand(5,1),[],[],[],[],zeros(5,1))
%
% function out = myobj(A,a,w,beta,lambda)
% resid = a - A*w;
% out = 0.5*norm(resid)^2 + (beta/2)*norm(w)^2 + lambda*sum(abs(w));

