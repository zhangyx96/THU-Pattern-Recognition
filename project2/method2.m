%2018/11/28 张元鑫 2018210902 模式识别实验2
%稀疏字典学习
clc;clear;close all;
Data = load('train_data2');
Data = double(Data.Data)';
[M,N]=size(Data);
K = 500;%原子数
L = 10; %稀疏维数
if K<N
    init = randperm(N,K); %随机挑选K个样本作为初始原子
else
    init = randi(N,K,1);
end
Dict = Data(:,init); %初始字典
for i=1:K
    Nor(1,i)=norm(Dict(:,i));  %归一化成单位向量 
end
Nor=repmat(Nor,M,1);
Dict=Dict./Nor;
epoch = 50;
loss = zeros(epoch,1);
for j =1:epoch
    x = OMP(Dict,Data,L); %OMP求稀疏化矩阵
    for k = 1:K
        k_index = find(x(k,:)); %找到稀疏系数
        if isempty(k_index)~=1 %判断稀疏系数为空的情况
            x_k=x(:,k_index);
            x_k(k,:)=0;
            y_k=Dict*x_k;
            y_t=Data(:,k_index);  %样本真值
            E_k=y_t-y_k;  %计算残差
            [U,S,V]=svds(E_k,1); %用SVD分解残差，保留特征值最大的（为提高收敛速度，这里没用自己写的SVD）
            Dict(:,k)=U; %更新字典
            x(k,k_index)=S*V'; %更新稀疏矩阵
        end
    end
    Data_C = Dict*x;
    E = (Data_C-Data).^2;
    loss(j) = log(sum(sum(E)));  %计算loss
    display(['[epoch',num2str(j),']','loss:',num2str(loss(j))]);
end
figure;  %画图
figure_title =['L=',num2str(L),' K=',num2str(K),' 均方误差收敛曲线'];
plot(1:epoch,loss);
title(figure_title);
saveas(gcf,figure_title,'bmp');
save_title = ['L=',num2str(L),' K=',num2str(K),'.mat'];
save(save_title,'loss');









