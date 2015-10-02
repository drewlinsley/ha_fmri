function fun_vis(rh_stats,lh_stats,sum_fun)

rh_stats_mu = sum_fun(cat(3,rh_stats{:}),3);
lh_stats_mu = sum_fun(cat(3,lh_stats{:}),3);

figure,
subplot(1,2,1),
imagesc(rh_stats_mu)
colorbar
title('Jonas is a Right-Hemisphere Gomer');
subplot(1,2,2),
imagesc(lh_stats_mu),
colorbar
title('Jonas can go fuck himself left-hemisphere')

figure,
for idx = 1:numel(lh_stats),
    subplot(5,2,idx)
    imagesc(lh_stats{idx}),colorbar,
    title(sprintf('Left Hemisphere %i',idx));
end

figure,
for idx = 1:numel(rh_stats),
    subplot(5,2,idx)
    imagesc(rh_stats{idx}),colorbar,
    title(sprintf('Right Hemisphere %i',idx));
end



