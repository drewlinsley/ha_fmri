function roi_data = get_roi_data(roi_vol,stats)
roi_data = cell(numel(roi_vol(1,1,:)),1);
for idx = 1:numel(roi_vol(1,1,:)),
    tvol = zeros(numel(stats(1,1,:)),sum(sum(roi_vol(:,:,idx)==1)));
    nt = numel(stats(1,1,:));
    if ~isempty(tvol),
        if nt > 1,
            for il = 1:numel(stats(1,1,:)),
                tf = stats(:,:,il);
                tvol(il,:) = tf(roi_vol(:,:,idx)==1);
            end
        else
            tvol = stats(roi_vol(:,:,idx)==1);
        end
    end
    roi_data{idx} = tvol;
end