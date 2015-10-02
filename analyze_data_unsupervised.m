%Run control analysis unsupervised learning
clear; close all; clc;
home_dir = '/Users/drewlinsley/Downloads/mmbl_data';
function_dir = fullfile(home_dir,'functions');
addpath(genpath(function_dir));
data_dir = fullfile(home_dir,'data');
image_dir = fullfile(home_dir,'images');
roi_dir = fullfile(home_dir,'roi_grids');
output_dir = fullfile(home_dir,'output');
if ~exist(output_dir,'dir'),
    mkdir(output_dir);
end

%Settings
tr_shift = 2;
run_networks = true;
p_cut = 0.05;
model_name = 'scene_model';%'flickr_model';%'scene_model';%

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

chosen_right = {'rsts_grid'}; %{'rppa_grid'}; %
chosen_left = {'lsts_grid'}; %{'lppa_grid'}; %


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
    rh_movies = normalize_data(files,run_ids,1,tr_shift,false); %last arg applies median filter from gallant
    lh_movies = normalize_data(files,run_ids,2,tr_shift,false);
    
    %Average movies
    rh_movies_mean = mean(cat(4,rh_movies{:}),4);
    lh_movies_mean = mean(cat(4,lh_movies{:}),4);
    
    %Reshape movies
    rh_movies_mean = reshape(rh_movies_mean,[],size(rh_movies_mean,3));
    lh_movies_mean = reshape(lh_movies_mean,[],size(lh_movies_mean,3));
    rh_nans = (isnan(sum(rh_movies_mean,2))); %keep a record
    lh_nans = (isnan(sum(lh_movies_mean,2)));
    rh_nans_ids = find(rh_nans); %keep a record
    lh_nans_ids = find(lh_nans);
    rh_movies_mean(rh_nans_ids,:) = [];
    lh_movies_mean(lh_nans_ids,:) = [];
    
    % Setup ICA:
    K = 10;
    n_random_initializations = 1e3;
    random_seed = 1;
    plot_figures = 0;
    
    % Run right hemi
    [r_Rica_r, r_Wica, r_Rpca, r_Wpca] = mutual_information_ICA(rh_movies_mean, K, n_random_initializations, random_seed, plot_figures);
    r_pcacheck = corr(rh_movies_mean,r_Rpca);
    r_icacheck = corr(rh_movies_mean,r_Rica_r);
    
    % Run left hemi
    [l_Rica_r, l_Wica, l_Rpca, l_Wpca] = mutual_information_ICA(lh_movies_mean, K, n_random_initializations, random_seed, plot_figures);
    l_pcacheck = corr(lh_movies_mean,l_Rpca);
    l_icacheck = corr(lh_movies_mean,l_Rica_r);
    
    save(fullfile(output_dir,sprintf('unsupervised_%s_%s_data',model_name,date)),'r_Rica_r','r_Wica',...
        'r_Rpca','r_Wpca','l_Rica_r','l_Wica','l_Rpca','l_Wpca','rh_nans','lh_nans','rh_movies','lh_movies');
else
    files = dir(fullfile(output_dir,'*.mat'));
    files = {files(:).name};
    files = sort_nat(files);
    load(fullfile(output_dir,files{end}));
end

%Put Ricas back
rh_nonan_ids = find(~rh_nans);
lh_nonan_ids = find(~lh_nans);
figure,
for ol = 1:K,
    rh_ica = zeros(size(rh_movies{1},1),size(rh_movies{1},2));
    lh_ica = zeros(size(lh_movies{1},1),size(lh_movies{1},2));
    for idx = 1:numel(rh_nonan_ids),
        rh_ica(rh_nonan_ids(idx)) = r_Rica_r(idx,ol);
    end
    subplot(2,10,ol),imagesc(rh_ica);title(sprintf('RH %i',ol));
    for idx = 1:numel(lh_nonan_ids),
        lh_ica(lh_nonan_ids(idx)) = l_Rica_r(idx,ol);
    end
    subplot(2,10,ol + 10),imagesc(lh_ica);title(sprintf('LH %i',ol));
end

%Compare Wicas to imagenet
model_name = 'jonas_object_model';%'flickr_model';%'scene_model';%
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
obj_rh_model_corrs = corr(r_Wica',trimmed_object_model);
obj_lh_model_corrs = corr(l_Wica',trimmed_object_model);


%Compare Wicas to places
model_name = 'scene_model';%'flickr_model';%'scene_model';%
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
place_rh_model_corrs = corr(r_Wica',trimmed_object_model);
place_lh_model_corrs = corr(l_Wica',trimmed_object_model);


%Compare Wicas to subitizing
model_name = 'subitizing_model';%'flickr_model';%'scene_model';%
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
sub_rh_model_corrs = corr(r_Wica',trimmed_object_model);
sub_lh_model_corrs = corr(l_Wica',trimmed_object_model);


%Compare Wicas to flickr
model_name = 'flickr_model';%'flickr_model';%'scene_model';%
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
flick_rh_model_corrs = corr(r_Wica',trimmed_object_model);
flick_lh_model_corrs = corr(l_Wica',trimmed_object_model);

%### Look at mean(,2) of all the corrs to get average performance for
%components versus each model.

%Save analysis stuff
save(fullfile(output_dir,sprintf('unsupervised_stuff_%s_analysis',date)),'flick_rh_model_corrs',...
    'flick_lh_model_corrs','sub_rh_model_corrs','sub_lh_model_corrs','place_rh_model_corrs',...
    'place_lh_model_corrs','obj_rh_model_corrs','obj_lh_model_corrs');
