%2018/11/28 ��Ԫ�� 2018210902 ģʽʶ��ʵ��2
%ϡ���ֵ�ѧϰ
clc;clear;close all;
Data = load('train_data2');
Data = double(Data.Data)';
[M,N]=size(Data);
K = 500;%ԭ����
L = 10; %ϡ��ά��
if K<N
    init = randperm(N,K); %�����ѡK��������Ϊ��ʼԭ��
else
    init = randi(N,K,1);
end
Dict = Data(:,init); %��ʼ�ֵ�
for i=1:K
    Nor(1,i)=norm(Dict(:,i));  %��һ���ɵ�λ���� 
end
Nor=repmat(Nor,M,1);
Dict=Dict./Nor;
epoch = 50;
loss = zeros(epoch,1);
for j =1:epoch
    x = OMP(Dict,Data,L); %OMP��ϡ�軯����
    for k = 1:K
        k_index = find(x(k,:)); %�ҵ�ϡ��ϵ��
        if isempty(k_index)~=1 %�ж�ϡ��ϵ��Ϊ�յ����
            x_k=x(:,k_index);
            x_k(k,:)=0;
            y_k=Dict*x_k;
            y_t=Data(:,k_index);  %������ֵ
            E_k=y_t-y_k;  %����в�
            [U,S,V]=svds(E_k,1); %��SVD�ֽ�в��������ֵ���ģ�Ϊ��������ٶȣ�����û���Լ�д��SVD��
            Dict(:,k)=U; %�����ֵ�
            x(k,k_index)=S*V'; %����ϡ�����
        end
    end
    Data_C = Dict*x;
    E = (Data_C-Data).^2;
    loss(j) = log(sum(sum(E)));  %����loss
    display(['[epoch',num2str(j),']','loss:',num2str(loss(j))]);
end
figure;  %��ͼ
figure_title =['L=',num2str(L),' K=',num2str(K),' ���������������'];
plot(1:epoch,loss);
title(figure_title);
saveas(gcf,figure_title,'bmp');
save_title = ['L=',num2str(L),' K=',num2str(K),'.mat'];
save(save_title,'loss');









