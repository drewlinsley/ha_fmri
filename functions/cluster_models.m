function out_data = cluster_models(in_model,strip_intercept)

out_data = cell(numel(in_model),1);
for idx = 1:numel(in_model), %iterate through masks
    x=in_model{idx}';
    if strip_intercept == true,
        x(:,1) = [];
    end
    N=size(x,1);
    M=N*N-N;
    s=zeros(M,3);
    j=1;
    for i=1:N
        for k=[1:i-1,i+1:N]
            s(j,1)=i;
            s(j,2)=k;
            s(j,3)=-sum((x(i,:)-x(k,:)).^2);
            j=j+1;
        end;
    end;
    p=median(s(:,3));
    out_data{idx}=apcluster(s,p);
end
