function data_cell = normalize_data(files,run_ids,wv,tr_shift,mf)
if ~exist('mf','var'),mf = false; end %median filtering from gallant 2013
data_cell = cell(numel(files),1);
for idx = 1:numel(files),
    load(files{idx})
    new_data = data{wv};
    fprintf('Working video %i\r',idx)
    for il = 1:numel(unique(run_ids)),
        for x = 1:numel(data{wv}(:,1,1)),
            for y = 1:numel(data{wv}(1,:,1)),
                if ~isnan(sum(data{wv}(x,y,run_ids==il))),
                    mean_pad = repmat(mean(squeeze(data{wv}(x,y,run_ids==il))),tr_shift,1);
                    it_data = squeeze(data{wv}(x,y,run_ids==il));
                    if mf == true,
                        %apply median filter
                        it_data = medfilt1(it_data,120); %gallant uses this so...
                    end
                    comb_data = cat(1,it_data(tr_shift+1:end),mean_pad);
                    comb_data = zscore(comb_data);
                    comb_data(comb_data > 3) = 3;
                    comb_data(comb_data < -3) = -3;
                    new_data(x,y,run_ids==il) = comb_data;
                end
            end
        end
    end
    data_cell{idx} = new_data;
end

