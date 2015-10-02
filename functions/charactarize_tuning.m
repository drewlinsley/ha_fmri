function roi_tuning = charactarize_tuning(model,clusters,labels,int)


for idx = 1:numel(model), %iterate through masks,
    tm = model{idx};
    tc = clusters{idx};
    if int == true,
        tm = tm(2:end,:);
    end
    uni_clusts = unique(tc);
    num_clusts = numel(uni_clusts);
    for il = 1:num_clusts,
        it_data = tm(:,tc == uni_clusts(il));
        [~,it_max] = max(abs(it_data),[],2);
        
        
    end
end

beep