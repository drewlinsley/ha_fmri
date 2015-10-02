

movie_cut_ids = find(movie_cuts);
movie_cut_ids(end+1) = numel(this_vox_1) + 1;
ftv1 = zeros(size(this_vox_1));
ftv2 = zeros(size(this_vox_2));
K.RT = 2;
K.HParam = 128;

for idx = 1:numel(movie_cut_ids) - 1,
    start_idx = movie_cut_ids(idx);
    end_idx = movie_cut_ids(idx + 1) - 1;
    K.row = numel(start_idx:end_idx);
    nK = spm_filter(K);
    Y = spm_filter(nK, convreg);
    ftv1(start_idx:end_idx) = zscore(this_vox_1(start_idx:end_idx));
    ftv2(start_idx:end_idx) = zscore(this_vox_2(start_idx:end_idx));
end

ftv1(ftv1 > 3) = 3;
ftv1(ftv1 < -3) = -3;
ftv2(ftv2 > 3) = 3;
ftv2(ftv2 < -3) = -3;
