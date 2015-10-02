zrh_1 = zscore(rh_1,0,3);
zrh_2 = zscore(rh_2,0,3);

zrh_1(zrh_1 > 3) = 3;
zrh_1(zrh_1 < -3) = -3;
zrh_2(zrh_2 > 3) = 3;
zrh_2(zrh_2 < -3) = -3;

rhs = size(rh_1);
cc = zeros(rhs(1),rhs(2),'single');
for x = 1:rhs(1),
    for y = 1:rhs(2),
        cc(x,y) = corr(squeeze(zrh_1(x,y,:)),squeeze(zrh_2(x,y,:)));
    end
end

hist(cc(:),50000)

figure,
plot(squeeze(zrh_1(50,100,:)))
hold on,plot(squeeze(zrh_2(50,100,:)))
figure,imagesc(cc)

