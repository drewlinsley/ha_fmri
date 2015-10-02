function [data_cells,selected_masks,mask_inds] = extract_roi_data(masks,input,mask_names,desired_masks)

data_cells = cell(numel(desired_masks),1);
selected_masks = data_cells;
mask_inds = data_cells;
for idx = 1:numel(desired_masks),
    wm = cellfun(@(x) ~isempty(x),cellfun(@(x) regexp(x,desired_masks{idx}),mask_names,'UniformOutput',false));
    tm = masks(:,:,wm);
    data_mat = zeros(size(input,3),sum(sum(tm == 1)));
    for il = 1:size(input,3),
        ts = input(:,:,il);
        tm_inds = find(tm==1);
        data_mat(il,:) = ts(tm_inds);
    end
    selected_masks{idx} = tm;
    data_cells{idx} = data_mat;
    mask_inds{idx} = tm_inds;
end
