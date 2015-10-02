function [roi_centroid,roi_outline,roi_vol] = overlay_rois(roi_files,hemisphere,data)

load(roi_files{1});
roi_vol = zeros(size(grid_roi_thresh{hemisphere},1),size(grid_roi_thresh{hemisphere},2),numel(roi_files),'single');
roi_centroid = zeros(numel(roi_files),2);
roi_outline = cell(numel(roi_files),1);
figure,hold on
imagesc(data);
colors = jet(numel(roi_files));
for idx = 1:numel(roi_files),
    load(roi_files{idx});
    roi_vol(:,:,idx) = grid_roi_thresh{hemisphere};
    rs = regionprops(roi_vol(:,:,idx) > 0,'Centroid','ConvexHull');
    if ~isempty(rs),
        roi_centroid(idx,:) = mean(cat(1,rs(:).Centroid));
        all_conv = cat(1,rs(:).ConvexHull);
        filt_conv = convhull(all_conv(:,1),all_conv(:,2));
        roi_outline{idx} = cat(2,all_conv(filt_conv,1),all_conv(filt_conv,2));
        plot(roi_outline{idx}(:,1),roi_outline{idx}(:,2),'Color',colors(idx,:))
        try
        tlab = regexp(roi_files{idx},'/?+');
        tlab = strrep(roi_files{idx}(tlab(end)+1:end),'_',' ');
        catch
            beep
        end
        text(roi_centroid(idx,1),roi_centroid(idx,2),tlab,'Color','white');
    end
end

