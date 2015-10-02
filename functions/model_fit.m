function stats = train_model_jonas(in_mod,in_mov,X,num_toss,folds)

if ~exist('folds','var'), folds = 10; end
if ~exist('lambda','var'), lambda = 10; end
if ~exist('wm','var'),wm = 1; end
if ~exist('num_toss','var'),num_toss = 5; end
input_inds = reshape(repmat(1:folds,ceil(numel(in_mov{1}(1,1,:))/folds),1),[],1);
if numel(input_inds) > numel(in_mov{1}(1,1,:)),
    input_inds = input_inds(1:numel(in_mov{1}(1,1,:)));
elseif numel(input_inds) < numel(in_mov{1}(1,1,:)),
    input_inds = cat(1,input_inds,repmat(input_inds(end),numel(in_mov{1}(1,1,:)) - numel(input_inds),1));
end

i_X = cat(2,ones(size(X,1),1),X);
stats = zeros(size(in_mov{1},1),size(in_mov{1},2),folds,'single');
for idx = 1:folds,
    fprintf('Working fold %i/%i\r',idx,folds);
    %Take each run, train on 80% and test on 20% (tossing the first 5%)
    training_ids = setdiff(1:folds,idx); %figure out training IDs
    training_inds = ismember(input_inds,training_ids);
    training_data = cellfun(@(x) x(:,:,training_inds),in_mov,'UniformOutput',false);
    training_X = i_X(training_inds,:);
    
    if wm == 1,
        %%Mean method
        train_Y = mean(cat(4,training_data{:}),4);
    elseif wm == 2,
        %%Cat method
        train_Y = cat(3,training_data{:});
        %
    end
    
    for x = 1:size(train_Y,1),
        for y = 1:size(train_Y,2),
            if ~isnan(sum(squeeze(train_Y(x,y,:)))),
                it_vtc = squeeze(train_Y(x,y,:));
                it_pred = training_X * squeeze(in_mod{idx}(x,y,:));
                it_corr = corr(it_vtc,it_pred);
                stats(x,y,idx) = it_corr;
            end
        end
    end
end

