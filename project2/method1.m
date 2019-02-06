%2018/11/28 张元鑫 2018210902 模式识别实验2
%SVD降维
clc;clear;close all;
Data = load('train_data2');
Data = double(Data.Data);
[M,N]=size(Data);
[U,S,V] = mySVD(Data);  %用自己编写的SVD函数进行SVD分解
K_list = [10 50 100];
for K = K_list
    U_k = U(:,1:K);
    S_k = S(1:K,1:K);
    V_k = V(:,1:K);
    Data_C = U_k*S_k*V_k';
    E = (Data_C-Data).^2; 
    loss = log(sum(sum(E)));
    display([num2str(K),':',num2str(loss)]);
end

