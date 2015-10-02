clear; close all; clc;
home_dir = '/Users/drewlinsley/Downloads/mmbl_data';
function_dir = fullfile(home_dir,'functions');
addpath(function_dir);
data_dir = fullfile(home_dir,'data');
image_dir = fullfile(home_dir,'images');
roi_dir = fullfile(home_dir,'roi_grids');
output_dir = fullfile(home_dir,'output');
if ~exist(output_dir,'dir'),
    mkdir(output_dir);
end

%Settings
tr_shift = 2;
run_networks = false;
p_cut = 0.05;
model_name = 'scene_model';%'object_model';%'subitizing_model';%'flickr_model';%'scene_model';%'jonas_object_model';%

%Start fun
files = {'movie_segment_run3_norm_grid','movie_segment_run4_norm_grid',...
    'movie_segment_run5_norm_grid','movie_segment_run6_norm_grid'};
%lrois = {'lh_A1_grid','lh_speech_grid','lhV1_grid','llo_grid','lofa_grid',...
%    'lpfs_grid','lppa_grid','lrsc_grid','lsts_grid','ltos_grid','notoverlap_lffa_grid'};
%rrois = {'rh_A1_grid','rh_speech_grid','rhV1_grid','rlo_grid','rofa_grid',...
%    'rpfs_grid','rppa_grid','rrsc_grid','rsts_grid','rtos_grid','rffa_grid'};
rrois = {'rh_A1_grid','rh_speech_grid','rhV1_grid','rlo_grid','rofa_grid',...
    'rpfs_grid','rppa_grid','rrsc_grid','rsts_grid','rtos_grid','rffa_grid',...
    'reba_grid','MD01_grid','MD02_grid','MD03_grid','MD04_grid','MD05_grid',...
    'MD06_grid','MD07_grid','MD08_grid','MD09_grid','MD10_grid','MD11_grid',...
    'MD12_grid','MD13_grid','MD14_grid','MD15_grid','MD16_grid','MD17_grid',...
    'MD18_grid','KeyLang01_grid','KeyLang02_grid','KeyLang03_grid','KeyLang04_grid',...
    'KeyLang05_grid','KeyLang06_grid','KeyLang07_grid','KeyLang08_grid',...
    'Articloc01_grid','Articloc02_grid','Articloc03_grid','Articloc04_grid',...
    'Articloc06_grid'};
lrois = {'lh_A1_grid','lh_speech_grid','lhV1_grid','llo_grid','lofa_grid','notoverlap_lvwfa_grid',...
    'notoverlap_lpfs_grid','lppa_grid','lrsc_grid','lsts_grid','ltos_grid','notoverlap_lffa_grid',...
    'leba_grid','MD01_grid','MD02_grid','MD03_grid','MD04_grid','MD05_grid',...
    'MD06_grid','MD07_grid','MD08_grid','MD09_grid','MD10_grid','MD11_grid',...
    'MD12_grid','MD13_grid','MD14_grid','MD15_grid','MD16_grid','MD17_grid',...
    'MD18_grid','KeyLang01_grid','KeyLang02_grid','KeyLang03_grid','KeyLang04_grid',...
    'KeyLang05_grid','KeyLang06_grid','KeyLang07_grid','KeyLang08_grid',...
    'Articloc01_grid','Articloc02_grid','Articloc03_grid','Articloc04_grid',...
    'Articloc06_grid'};

chosen_right = {'rffa_grid'}; %{'rsts_grid'}; %{'rppa_grid'}; %{'rtos_grid'};%{'rffa_grid'};%
chosen_left = {'notoverlap_lffa_grid'}; %{'lsts_grid'}; %{'lppa_grid'}; %{'ltos_grid'};%{'notoverlap_lffa_grid'};%


%Sts, EBA, OFA, LO, EVC
%rrois = {'rsts_grid'};
%lrois = {'lsts_grid'};

rrois = cellfun(@(x) fullfile(roi_dir,x),rrois,'UniformOutput',false);
lrois = cellfun(@(x) fullfile(roi_dir,x),lrois,'UniformOutput',false);
files = cellfun(@(x) fullfile(data_dir,x),files,'UniformOutput',false);
run_ids = cat(2,ones(1,254),ones(1,533 - 254) + 1);

%Get old movie times
load(fullfile(data_dir,'ha_data'),'movie_times')
old_movie_times = movie_times;
%Get new movie times
load(files{1},'movie_times')
movie_times_idx = ismember(old_movie_times,movie_times);
%Load conv features
%load(fullfile(data_dir,'object_model'))
load(fullfile(data_dir,model_name))
trimmed_object_model = object_model(movie_times_idx,:);
%Load conv labels
if strcmp(model_name,'scene_model'),
    w2v_scene_labels;
elseif strcmp(model_name,'object_model'),
    w2v_object_labels;
else
    model_labels = readtable(fullfile(data_dir,strcat(model_name,'_labels')));
    fn = fieldnames(model_labels);
    model_labels = model_labels.(fn{1});
end
%Get pointers for images
im_names = dir(fullfile(image_dir,model_name,'*.jpg'));
im_names = cellfun(@(x) fullfile(image_dir,model_name,x),cat(1,{im_names(:).name}),'UniformOutput',false);

if run_networks == true,
    %Run/voxel-wise normalize data
    rh_movies = normalize_data(files,run_ids,1,tr_shift); %MAKE SURE THIS IS LAGGING DATA!!!!
    lh_movies = normalize_data(files,run_ids,2,tr_shift);
    
    %Check 1-run held out voxel correlations
    rh_movies_cc = check_cross_run_corrs(rh_movies);
    lh_movies_cc = check_cross_run_corrs(lh_movies);
    
    %Plot each
    display_cc(rh_movies_cc,'right');
    display_cc(lh_movies_cc,'left');
        
    %Train models at each voxel
    [rh_stats,rh_models,rh_train_stats,rh_train_lambda,rh_pvals] = train_model_jonas(rh_movies,trimmed_object_model,3,[.01,.1,1,10],10);
    [lh_stats,lh_models,lh_train_stats,lh_train_lambda,lh_pvals] = train_model_jonas(lh_movies,trimmed_object_model,3,[.01,.1,1,10],10); %options are #trim, lambda, folds, and null test
    
    save(fullfile(output_dir,sprintf('2roi_%s_%s_data',model_name,date)),'rh_stats','rh_models',...
        'lh_stats','lh_models','rh_train_stats','rh_train_lambda',...
        'lh_train_stats','lh_train_lambda',...
        'lh_movies_cc','rh_movies_cc','rh_movies','lh_movies','rh_pvals','lh_pvals');
else
    mat_files = dir(fullfile(output_dir,strcat('*',model_name,'*')));
    mat_files = {mat_files(:).name};
    trunc = cellfun(@(x) ~isempty(x),cellfun(@(x) regexp(x,'data'),mat_files,'UniformOutput',false));
    mat_files = mat_files(trunc);
    load(fullfile(output_dir,mat_files{1}));
end

%Visualize stats and rois
fun_vis(rh_stats,lh_stats,@mean);
[~,~,r_roi_vol_masks] = overlay_rois(rrois,1,rh_stats{end});
[~,~,l_roi_vol_masks] = overlay_rois(lrois,2,lh_stats{end});

%Extract ROI data
[r_roi_vol_fits,r_selected_masks,r_mask_inds] = extract_roi_data(r_roi_vol_masks,rh_stats{end},rrois,chosen_right);
r_roi_vol_models = extract_roi_data(r_roi_vol_masks,rh_models{end},rrois,chosen_right);
[l_roi_vol_fits,l_selected_masks,l_mask_inds] = extract_roi_data(l_roi_vol_masks,lh_stats{end},lrois,chosen_left);
l_roi_vol_models = extract_roi_data(l_roi_vol_masks,lh_models{end},lrois,chosen_left);

%Split the requested ROIs into clusters
r_roi_vol_clusters = cluster_models(r_roi_vol_models,true); %true strips the intercept
l_roi_vol_clusters = cluster_models(l_roi_vol_models,true); 

%Take a look at how the weights cluster
figure,subplot(4,1,1),plot(r_roi_vol_fits{1}),title(sprintf('%s',chosen_right{1})),subplot(4,1,2),xlabel('Voxels'),ylabel('Flickr'),
imagesc(r_roi_vol_models{1}(2:end,:)),subplot(4,1,3),stem(r_roi_vol_clusters{1});
subplot(4,1,4),dendrogram(linkage(r_roi_vol_models{1}(2:end,:)','ward'),0);
figure,subplot(4,1,1),plot(l_roi_vol_fits{1}),title(sprintf('%s',chosen_left{1})),subplot(4,1,2),xlabel('Voxels'),ylabel('Flickr'),
imagesc(l_roi_vol_models{1}(2:end,:)),subplot(4,1,3),stem(l_roi_vol_clusters{1});
subplot(4,1,4),dendrogram(linkage(l_roi_vol_models{1}(2:end,:)','ward'),0);

%MC test for rois ONLY CONTINUE ANALYZING SUPRATHRESHOLD VOXELS
%for idx = 1:numel(chosen_right),
r_roi_pvals = hyp_test_fits(r_selected_masks{1},rh_movies,trimmed_object_model,3,[.01,.1,1,10],10);
r_pv_mask = (r_roi_pvals{end}(r_roi_pvals{end} < 1)) < p_cut;
%end
%for idx = 1:numel(chosen_left),
l_roi_pvals = hyp_test_fits(l_selected_masks{1},lh_movies,trimmed_object_model,3,[.01,.1,1,10],10);
l_pv_mask = (l_roi_pvals{end}(l_roi_pvals{end} < 1)) < p_cut;
%end

%Fix masks
r_selected_masks = pval_fix_mask(r_pv_mask,r_selected_masks);
r_mask_inds = pval_fix_mask(r_pv_mask,r_mask_inds);
r_roi_vol_clusters = pval_fix_mask(r_pv_mask,r_roi_vol_clusters);
r_roi_vol_models = pval_fix_mask(r_pv_mask,r_roi_vol_models);
l_selected_masks = pval_fix_mask(l_pv_mask,l_selected_masks);
l_mask_inds = pval_fix_mask(l_pv_mask,l_mask_inds);
l_roi_vol_clusters = pval_fix_mask(l_pv_mask,l_roi_vol_clusters);
l_roi_vol_models = pval_fix_mask(l_pv_mask,l_roi_vol_models);

%Get correlation ceiling
r_corr_ceiling = calculate_corr_ceiling(rh_movies,r_selected_masks);
l_corr_ceiling = calculate_corr_ceiling(lh_movies,l_selected_masks);
r_corr_ceiling_max = max(cat(2,r_corr_ceiling{end}{:}),[],2);
l_corr_ceiling_max = max(cat(2,l_corr_ceiling{end}{:}),[],2);


%Classify the category most likely indicated by each cluster at every
%timepoint -- Need to make sure these map to conv labels correctly.
r_roi_guesses = classify_cluster_timecourse(rh_movies,trimmed_object_model,model_labels,r_selected_masks,r_roi_vol_clusters);
l_roi_guesses = classify_cluster_timecourse(lh_movies,trimmed_object_model,model_labels,l_selected_masks,l_roi_vol_clusters);
r_roi_cms = view_cms(r_roi_guesses,10,model_labels);
l_roi_cms = view_cms(l_roi_guesses,10,model_labels);

%Look at spatial clustering
r_cluster_brain_inds = vis_clusters(rh_stats,r_mask_inds,r_roi_vol_clusters);
l_cluster_brain_inds = vis_clusters(lh_stats,l_mask_inds,l_roi_vol_clusters);

%Characterize cluster tuning
r_roi_tuning = characterize_tuning(r_roi_vol_models,r_roi_vol_clusters,model_labels,im_names,true,@(x) x); %2nd to last arg indicates an intercept on the model
l_roi_tuning = characterize_tuning(l_roi_vol_models,l_roi_vol_clusters,model_labels,im_names,true,@(x) x);
r_roi_tuning_cukor = characterize_tuning_cukor(rh_movies,r_selected_masks,trimmed_object_model,r_roi_vol_clusters,model_labels,im_names); %2nd to last arg indicates an intercept on the model
l_roi_tuning_cukor = characterize_tuning_cukor(lh_movies,l_selected_masks,trimmed_object_model,l_roi_vol_clusters,model_labels,im_names);

%save extra stuff
save(fullfile(output_dir,sprintf('2roi_%s_%s_analysis',model_name,date)),'r_roi_vol_masks','l_roi_vol_masks',...
    'r_roi_vol_fits','l_roi_vol_fits','r_selected_masks','l_selected_masks',...
    'r_roi_vol_clusters','l_roi_vol_clusters','r_roi_guesses','l_roi_guesses',...
    'r_roi_tuning','l_roi_tuning','r_corr_ceiling','l_corr_ceiling','r_mask_inds',...
    'r_mask_inds','l_mask_inds','r_cluster_brain_inds','l_cluster_brain_inds',...
    'r_roi_cms','l_roi_cms','r_roi_pvals','l_roi_pvals','r_roi_vol_models','l_roi_vol_models',...
    'r_pv_mask','l_pv_mask','r_roi_tuning_cukor','l_roi_tuning_cukor');


%Compare classifications to actual video
if ~exist(fullfile(output_dir,'trimmed_HA.mat'),'file'),
    trimmed_movie = get_movie_frames(fullfile(data_dir,'Home_Alone_2_PG.m4v'),movie_times);
    save(fullfile(output_dir,'trimmed_HA.mat'),'trimmed_movie');
else
    load(fullfile(output_dir,'trimmed_HA.mat'))
end
r_roi_movie = guesses_vs_video(r_roi_guesses,trimmed_object_model,trimmed_movie,im_names,model_labels,5);
l_roi_movie = guesses_vs_video(l_roi_guesses,trimmed_object_model,trimmed_movie,im_names,model_labels,3);
