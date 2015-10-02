
load(fullfile(data_dir,'scene_word_data.mat'))
    model_labels = readtable(fullfile(data_dir,strcat(model_name,'_labels')));
    fn = fieldnames(model_labels);
    model_labels = model_labels.(fn{1});

x = word_data;
N=size(x,1);
M=N*N-N;
s=zeros(M,3);
j=1;
for i=1:N
for k=[1:i-1,i+1:N]
s(j,1)=i;
s(j,2)=k;
s(j,3)=-sum((x(i,:)-x(k,:)).^2);
j=j+1;
end;
end;
p=median(s(:,3));
new_labels=apcluster(s,p);
uni_labels = unique(new_labels);
num_labels = numel(uni_labels);
nc = cell(num_labels,1);
for i = 1:num_labels,
    nc{i} = model_labels(new_labels==uni_labels(i));
end
model_labels = cellfun(@(x) x{1},nc,'UniformOutput',false);

nom = zeros(size(trimmed_object_model,1),num_labels);
for iii = 1:num_labels,
    nom(:,iii) = max(trimmed_object_model(:,new_labels == uni_labels(iii)),[],2);
end
trimmed_object_model = nom;    



return
%handpicked labels

model_labels = {'airport','outdoor_manmade','outdoor_natural',...
    'sportsfield','car','coast','church','small_med_scene',...
    'engine_room','gas_station','highway','martial_arts_gym',...
    'outdoor_church','ocean','active_scene','serene_outdoor',...
    'rock_arch','sea_cliff','television_studio'}; %gathered by clustering word2vec place names w/ AP
