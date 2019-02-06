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
        dist=D'*res; %����������ԭ��
        [~,pos]=max(abs(dist));
        indx(j)=pos;
        a=pinv(D(:,indx(1:j)))*y; %��С���˷���x�Ĺ���ֵ
        res=y-D(:,indx(1:j))*a; %����в�
        if sum(res.^2) < rH %��ǰ��ֹ,����ٶ�
            break;
        end
    end;
    output=zeros(K,1);
    output(indx(1:j))=a;
    X(:,i)=sparse(output); %ϡ����
end;
return;
