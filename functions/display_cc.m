function display_cc(input,label)
sf = factor(numel(input));
figure,
for idx = 1:numel(input),
    subplot(sf(1),sf(2),idx);
    imagesc(input{idx})
    colorbar
    title(sprintf('Input %i of %s',idx,label))
end