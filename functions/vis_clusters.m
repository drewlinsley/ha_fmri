function brain_inds = vis_clusters(stats,inds,clusters)


%outer loop is CV
brain_inds = cell(numel(stats),1);
for idx = 1:numel(stats),
    if ~isempty(stats{idx}),
        %inner loop 1 is masks
        for il = 1:numel(inds),
            %inner loop 2 is clusters
            ti = inds{il};
            tc = clusters{il};
            uni_clusts = unique(tc);
            num_clusts = numel(uni_clusts);
            fix_clusts = tc;
            for iil = 1:num_clusts,
                sel = fix_clusts == uni_clusts(iil);
                fix_clusts(sel) = iil;
            end
            fix_inds = cat(2,ti,fix_clusts);
            brain_vol = abs(stats{idx});
            brain_vol = repmat(brain_vol,1,1,3);
            
            colors = jet(num_clusts);
            for iil = 1:numel(fix_inds(:,1)),
                for ci = 1:size(brain_vol,3),
                    cs = brain_vol(:,:,ci);
                    thc = colors(fix_inds(iil,2),ci);
                    cs(fix_inds(iil,1)) = thc;
                    brain_vol(:,:,ci) = cs;
                end
            end
            figure,
            imagesc(brain_vol)
            brain_inds{idx}{il} = brain_inds;
        end
    end
end
