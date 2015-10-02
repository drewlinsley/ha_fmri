function preds = classify_cluster_timecourse(input,X,labels,masks,clusters,folds,wm,num_toss)

if ~exist('folds','var'), folds = 10; end
if ~exist('wm','var'),wm = 1; end
if ~exist('num_toss','var'),num_toss = 3; end

input_inds = reshape(repmat(1:folds,ceil(numel(X(:,1))  /folds),1),[],1);
if numel(input_inds) > numel(input{1}(1,1,:)),
    input_inds = input_inds(1:numel(input{1}(1,1,:)));
elseif numel(input_inds) < numel(input{1}(1,1,:)),
    input_inds = cat(1,input_inds,repmat(input_inds(end),numel(input{1}(1,1,:)) - numel(input_inds),1));
end

warning off
preds = cell(numel(folds),1);
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
        train_Y = mean(cat(4,training_data{:}),4);
        test_Y = mean(cat(4,testing_data{:}),4);
        train_X = X(training_inds,:);
        test_X = X(testing_inds,:);
        [~,hard_train_X] = max(train_X,[],2);
        [~,hard_test_X] = max(test_X,[],2);
    elseif wm == 2,
        %%Cat method
        test_Y = cat(3,testing_data{:});
        train_Y = cat(3,training_data{:});
    end

    %Iterate through masks, only preserving mask voxels
    mask_its = cell(numel(masks),1);
    for il = 1:numel(masks),
        train_Y_mat = zeros(sum(sum(masks{il}==1)),size(train_Y,3));
        for iil = 1:size(train_Y,3), %First do training data
            it_data = train_Y(:,:,iil);
            train_Y_mat(:,iil) = it_data(masks{il} == 1);
        end
        test_Y_mat = zeros(sum(sum(masks{il}==1)),size(test_Y,3));
        for iil = 1:size(test_Y,3), %First do training data
            it_data = test_Y(:,:,iil);
            test_Y_mat(:,iil) = it_data(masks{il} == 1);
        end
        %Iterate through clusters, doing classification along the way
        it_clusters = clusters{il};
        uni_ids = unique(it_clusters);
        it_mask_cluster_stats = cell(numel(uni_ids),1);
        for iil = 1:numel(uni_ids),
            it_train_Y_mat = train_Y_mat(it_clusters == uni_ids(iil),:)';
            it_test_Y_mat = test_Y_mat(it_clusters == uni_ids(iil),:)';
            it_mask_cluster_stats{iil} = softmax_classify(it_train_Y_mat,it_test_Y_mat,hard_train_X,hard_test_X,true);
        end
        mask_its{il} = it_mask_cluster_stats;
    end
    preds{idx} = mask_its;
end