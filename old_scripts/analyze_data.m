clear; close all; clc;
home_dir = '/Users/drewlinsley/Downloads/mmbl_data';
function_dir = fullfile(home_dir,'functions');
addpath(function_dir);
data_dir = fullfile(home_dir,'data');
roi_dir = fullfile(home_dir,'roi_grids');
output_dir = fullfile(home_dir,'output');
if ~exist(output_dir,'dir'),
    mkdir(output_dir);
end

%Settings
tr_shift = 2;
run_networks = true;
p_cut = 0.05;

%Start fun
files = {'movie_segment_run3_norm_grid','movie_segment_run4_norm_grid',...
    'movie_segment_run4_norm_grid','movie_segment_run6_norm_grid'};
%lrois = {'lh_A1_grid','lh_speech_grid','lhV1_grid','llo_grid','lofa_grid',...
%    'lpfs_grid','lppa_grid','lrsc_grid','lsts_grid','ltos_grid','notoverlap_lffa_grid'};
%rrois = {'rh_A1_grid','rh_speech_grid','rhV1_grid','rlo_grid','rofa_grid',...
%    'rpfs_grid','rppa_grid','rrsc_grid','rsts_grid','rtos_grid','rffa_grid'};

rrois = {'rlo_grid','rppa_grid'};


lrois = cellfun(@(x) fullfile(roi_dir,x),lrois,'UniformOutput',false);
rrois = cellfun(@(x) fullfile(roi_dir,x),rrois,'UniformOutput',false);
files = cellfun(@(x) fullfile(data_dir,x),files,'UniformOutput',false);
run_ids = cat(2,ones(1,254),ones(1,533 - 254) + 1);

%Get old movie times
load(fullfile(data_dir,'ha_data'),'movie_times')
old_movie_times = movie_times;
%Get new movie times
load(files{1},'movie_times')
movie_times_idx = ismember(old_movie_times,movie_times);

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
    
    %Load conv features
    load(fullfile(data_dir,'object_model'))
    trimmed_object_model = object_model(movie_times_idx,:);
    
%     %Train models at each voxel
%     [rh_stats,rh_models] = train_model_jonas(rh_movies,trimmed_object_model,3,[0.1,1,10,100,1000]);
%     [lh_stats,lh_models] = train_model_jonas(lh_movies,trimmed_object_model,3,[0.1,1,10,100,1000]); %last argument is # of trials to trim pre/post testing
    %Train models at each roi
    [~,~,r_roi_vol] = overlay_rois(rrois,1,rh_movies_cc{1});
    [~,~,l_roi_vol] = overlay_rois(lrois,2,lh_movies_cc{1});
    all_r_rois = sum(r_roi_vol,3) > 0;
    all_l_rois = sum(l_roi_vol,3) > 0;
    [rh_stats,rh_models] = train_model_jonas_ROI(rh_movies,trimmed_object_model,all_r_rois,3,[0.1,1,10,100,1000]);
    [lh_stats,lh_models] = train_model_jonas_ROI(lh_movies,trimmed_object_model,all_l_rois,3,[0.1,1,10,100,1000]); %last argument is # of trials to trim pre/post testing
       
    save(fullfile(output_dir,sprintf('%s_data',date)),'rh_stats','lh_stats','rh_models','lh_models',...
        'rh_movies_cc','lh_movies_cc','rh_movies','lh_movies')
else
    files = dir(fullfile(output_dir,'*.mat'));
    files = {files(:).name};
    files = sort_nat(files);
    load(fullfile(output_dir,files{end}));
end
fun_vis(rh_stats,lh_stats,@median);

[r_roi_centroid,r_roi_outline,r_roi_vol] = overlay_rois(rrois,1,rh_stats{1});
[l_roi_centroid,l_roi_outline,l_roi_vol] = overlay_rois(lrois,2,lh_stats{1});
r_roi_fits = get_roi_data(r_roi_vol,rh_stats{1}); %last arg is which cv to use
l_roi_fits = get_roi_data(l_roi_vol,lh_stats{1}); %last arg is which cv to use
r_roi_weights = get_roi_data(r_roi_vol,rh_models{1}); %last arg is which cv to use
l_roi_weights = get_roi_data(l_roi_vol,lh_models{1}); %last arg is which cv to use

wb_rh_fit = mean(cat(3,rh_stats{:}),3)./std(cat(3,rh_stats{:}),0,3);
wb_lh_fit = mean(cat(3,lh_stats{:}),3)./std(cat(3,lh_stats{:}),0,3);
wb_rh_mask = wb_rh_fit > (icdf('normal',1-(p_cut*2),0,1));
wb_lh_mask = wb_lh_fit > (icdf('normal',1-(p_cut*2),0,1));
overlay_rois(rrois,1,wb_rh_mask);
overlay_rois(lrois,1,wb_lh_mask);


figure,imagesc(wb_rh_fit > 1.5)
figure,imagesc(wb_lh_fit > 1.5);



[x,y] = ind2sub(size(rh_stats{1}),find(rh_stats{1} == min(min(rh_stats{1}))));
true_tc = squeeze(rh_movies{1}(x,y,:));
pred_tc = cat(2,ones(size(trimmed_object_model,1),1),trimmed_object_model) * squeeze(rh_models{1}(x,y,:));

figure,
plot(true_tc,'b'),hold on,plot(pred_tc,'r')
title('worst')

[x,y] = ind2sub(size(rh_stats{1}),find(rh_stats{1} == max(max(rh_stats{1}))));
true_tc = squeeze(rh_movies{1}(x,y,:));
pred_tc = cat(2,ones(size(trimmed_object_model,1),1),trimmed_object_model) * squeeze(rh_models{1}(x,y,:));

figure,
plot(true_tc,'b'),hold on,plot(pred_tc,'r')
title('best')



rh_stats = model_fit(rh_models,rh_movies,trimmed_object_model,5,5);
