function X=OMP(D,Y,L)
[~,N]=size(Y);
[~,K]=size(D);
X = zeros([K,N]);
a = [];
rH = 1e-6; 
for i=1:N
    y=Y(:,i);
    res=y;
    indx=zeros(L,1);
    for j=1:L
        dist=D'*res; %求距离最近的原子
        [~,pos]=max(abs(dist));
        indx(j)=pos;
        a=pinv(D(:,indx(1:j)))*y; %最小二乘法求x的估计值
        res=y-D(:,indx(1:j))*a; %计算残差
        if sum(res.^2) < rH %提前终止,提高速度
            break;
        end
    end;
    output=zeros(K,1);
    output(indx(1:j))=a;
    X(:,i)=sparse(output); %稀疏表达
end;
return;
