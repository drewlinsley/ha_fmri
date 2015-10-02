function output = guesses_vs_video(roi_guesses,model,movie,images,labels,num_disp)

if ~exist('folds','var'), folds = 10; end
if ~exist('lambda','var'), lambda = 10; end
if ~exist('lambda_folds','var'),lambda_folds = 10; end
if ~exist('wm','var'),wm = 1; end
if ~exist('num_toss','var'),num_toss = 3; end
if ~exist('null','var'),null = false; end
if ~exist('num_iter','var'),num_iter = 200; end

input_inds = reshape(repmat(1:folds,ceil(numel(model(:,1))/folds),1),[],1);
if numel(input_inds) > size(model,1),
    input_inds = input_inds(1:size(model,1));
elseif numel(input_inds) < size(model,1),
    input_inds = cat(1,input_inds,repmat(input_inds(end),size(model,1) - numel(input_inds),1));
end
input_inds = input_inds == max(input_inds);
ii = find(input_inds);
ii(1:num_toss) = [];
ii(end - num_toss + 1 : end) = [];
input_inds = zeros(size(input_inds));
input_inds(ii) = 1;
input_inds = input_inds > 0;
movie = movie(:,:,:,input_inds);
[~,model_labels] = max(model,[],2);
model_labels = model_labels(input_inds);
trim_model = model(input_inds,:);
output = cell(numel(roi_guesses),1);
for idx = 1:numel(roi_guesses),
    tg = roi_guesses{idx};
    mask_holder = cell(numel(tg),1);
    if ~isempty(tg),
        %loop through masks
        for il = 1:numel(tg),
            tm = tg{il};
            %loop through clusters
            clust_holder = cell(numel(tm),1);
            for iil = 1:numel(tm),
                tc = tm{iil};
                if tc.acc_pval < 0.05, %let's look at sig classifiers
%                     [best_scores,best_ids] = sort(tc.scores,'descend');
%                     best_scores = best_scores(1:num_disp);
%                     best_ids = best_ids(1:num_disp);
                    lin_vec = floor(linspace(1,numel(tc.scores),num_disp));
                    best_scores = tc.scores(lin_vec);
                    best_ids = lin_vec;
                    
                    clust_holder{iil}.best_scores = best_scores;
                    clust_holder{iil}.best_ids = best_ids;
                    figure,
                    for mi = 1:num_disp,
                        subplot(2,num_disp,mi),
                        bp = tc.pred_labels(best_ids(mi));
                        clust_holder{iil}.best_frame(:,:,:,mi) = movie(:,:,:,best_ids(mi));
                        clust_holder{iil}.brain_prediction = tc.pred_labels;
                        clust_holder{iil}.chosen_frame_ids = best_ids;
                        clust_holder{iil}.model_prediction = model_labels;
                        ms = trim_model(best_ids(mi),:);
                        ms = -exp(ms) ./ sum(-exp(ms));
                        clust_holder{iil}.model_scores(mi) = max(ms);
                        
                        imshow(clust_holder{iil}.best_frame(:,:,:,mi)),
                        mp = model_labels(best_ids(mi));
                        title(sprintf('Model prediction %s',labels{mp}))
                        subplot(2,num_disp,mi + num_disp),
                        %imagesc(imread(images{bp}));
                        title(sprintf('Brain prediction %s, score = %.2f',labels{bp},best_scores(mi)))
                    end
                end
            end
            mask_holder{il} = clust_holder;
        end
    end
    output{idx} = mask_holder;
end