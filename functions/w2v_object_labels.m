model_labels = {'tiger';'electric';'bird';'frog';'lizard';'snake';'bug';'duck';...
    'alligator';'albatross';'dog';'fox';'cat';'beetle';'goldfish';'cricket';...
    'butterfly';'spider';'bison';'ram';'primate';'fish';'music';'baseball';...
    'beach';'beer';'sport';'car';'cassette';'boat';'cellular';'china';'holiday';...
    'coffee';'analog';'great';'disk';'drum';'file';'work';'freight';'health';'gas';...
    'grand';'greenhouse';'half';'gun';'honeycomb';'horizontal';'iron';'jean';'jersey';...
    'cook';'library';'magnetic';'medicine';'microwave';'military';'milk';'missile';...
    'calculator';'motor';'mouse';'acoustic';'oil';'oxygen';'dress';'park';'police';...
    'prayer';'mud';'radio';'school';'robin';'seat';'basketball';'solar';'steam';...
    'temple';'submarine';'hammerhead';'backpack';'throne';'aircraft';'bus';...
    'croquet';'wing';'web';'comic';'hot';'folding';'cup';'orange';'hip'}; %gathered by clustering word2vec place names w/ AP

load(fullfile(data_dir,'object_word_data.mat'))
% x = word_data;
% N=size(x,1);
% M=N*N-N;
% s=zeros(M,3);
% j=1;
% for i=1:N
% for k=[1:i-1,i+1:N]
% s(j,1)=i;
% s(j,2)=k;
% s(j,3)=-sum((x(i,:)-x(k,:)).^2);
% j=j+1;
% end;
% end;
% p=median(s(:,3));
% new_labels=apcluster(s,p);
uni_labels = unique(new_labels);
num_labels = numel(uni_labels);
% nc = cell(num_labels,1);
% for i = 1:num_labels,
% nc{i} = model_labels(new_labels==uni_labels(i));
% end



nom = zeros(size(trimmed_object_model,1),num_labels);
for iii = 1:num_labels,
    nom(:,iii) = max(trimmed_object_model(:,new_labels == uni_labels(iii)),[],2);
end
trimmed_object_model = nom;    