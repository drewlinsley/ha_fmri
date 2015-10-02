function preds = calculate_corr_ceiling(input,masks,num_its,folds,wm,num_toss)

if ~exist('num_its','var'), num_its = 200; end %number of resamples
if ~exist('folds','var'), folds = 10; end
if ~exist('wm','var'),wm = 1; end
if ~exist('num_toss','var'),num_toss = 3; end

input_inds = reshape(repmat(1:folds,ceil(size(input{1},3)/folds),1),[],1);
if numel(input_inds) > numel(input{1}(1,1,:)),
    input_inds = input_inds(1:numel(input{1}(1,1,:)));
elseif numel(input_inds) < numel(input{1}(1,1,:)),
    input_inds = cat(1,input_inds,repmat(input_inds(end),numel(input{1}(1,1,:)) - numel(input_inds),1));
end

preds = cell(numel(folds),1);
%for idx = 1:folds,
for idx = folds:folds,
    fprintf('Working fold %i/%i\r',idx,folds);
    testing_inds = ismember(input_inds,idx);    
    testing_data = cellfun(@(x) x(:,:,testing_inds),input,'UniformOutput',false);        
    cv_corrs = cell(numel(input),numel(masks));
    for il = 1:numel(input),
        target_data = testing_data{il};
        lo_data = testing_data;
        for mil = 1:numel(masks),
            test_Y_mat = extract_mask_data(target_data,masks,mil);
            corr_mat = zeros(size(test_Y_mat,2),numel(lo_data));
            for iil = 1:numel(lo_data),
                train_Y_mat = extract_mask_data(lo_data{iil},masks,mil);
                corr_mat(:,iil) = diag(corr(train_Y_mat,test_Y_mat));
            end
            corr_mat(:,il) = [];
            corr_mat = max(corr_mat,[],2); %max corr
            cv_corrs{il,mil} = corr_mat;
        end
    end
    preds{idx} = cv_corrs;
end

function output = extract_mask_data(input,masks,mil)
output = zeros(size(input,3),sum(sum(masks{mil}==1)));
for miil = 1:size(input,3),
    it_data = input(:,:,miil);
    output(miil,:) = it_data(masks{mil} == 1);
end