function data = pval_fix_mask(mask,data)


if iscell(data),
    for idx = 1:numel(data),
        td = data{idx};
        
        if iscell(td),
            for il = 1:numel(td),
                beep
            end
        else
            if size(td,1) == size(mask,1) && size(td,2) == size(mask,2),
                td(mask==0) = [];
                data{idx} = td;
            elseif size(td,2) == size(mask,1),
                td(:,mask==0) = [];
                data{idx} = td;
            else
                id_voxs = find(td == 1);
                id_voxs(mask==0) = [];
                new_td = zeros(size(td));
                new_td(id_voxs) = 1;
                data{idx} = new_td;
            end
        end
        
    end

else
    beep
end

