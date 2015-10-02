function cc = check_cross_run_corrs(input)

ns = size(input{1});
cc = cell(numel(input),1);
current_corr = zeros(ns(1),ns(2),'single');
for idx = 1:numel(input),
    fprintf('Working Input %i\r',idx)
    test_input = idx;
    other_inputs = input(setdiff(1:numel(input),idx));
    for il = 1:numel(other_inputs),
        for x = 1:ns(1),
            for y = 1:ns(2),
                Y = squeeze(input{test_input}(x,y,:));
                if ~isnan(sum(Y)),
                    X = cellfun(@(a) squeeze(a(x,y,:)),other_inputs,'UniformOutput',false);
                    X = cat(2,X{:});
                    current_corr(x,y) = mean(corr(Y,X)); %average cross-val corr
                end
            end
        end
    end
    cc{idx} = current_corr;
end
