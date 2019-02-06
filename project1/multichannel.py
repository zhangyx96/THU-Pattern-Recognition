import numpy as np
import scipy.io as sio 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def Gauss_predict(x,mu,var):
    
    px = np.zeros(len(mu))
    for i in range(len(mu)):    
        px[i] = -0.5*np.log(np.linalg.det(var[i,:,:]))
        px[i] += -0.5*((x-mu[i]).reshape(1,-1)).dot(np.linalg.inv(var[i,:,:])).dot((x-mu[i]).reshape(-1,1))
    return px

def _Gauss_predict(x,mu,sigma):
    joint_log_likelihood = np.zeros(len(mu))
    for i in range(len(mu)):
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * sigma[i, :]))
        n_ij -= 0.5 * np.sum(((x - mu[i]) ** 2) /(sigma[i, :]))
        joint_log_likelihood[i] = n_ij
    return np.argmax(joint_log_likelihood)

def get_mean_var(X,Y,mu,var,n_classes):
    for i in range(n_classes):
        l_index = np.where(Y==i+1)[0]
        X_i = X[l_index,:]        
        mu[i,:]= np.average(X_i,axis=0)    
        var[i,:,:] = np.cov(X_i.T)  



if __name__ == "__main__":
    
    data_path = 'D:/Office/研究生课程/模式识别/project1/data/'
    data_list = ['labels_name.mat','test_data.mat','train_data.mat']
    labels_name = sio.loadmat(data_path+data_list[0])
    train_data = sio.loadmat(data_path+data_list[1])
    test_data = sio.loadmat(data_path+data_list[2])
    train_x = np.array(train_data['Data'])
    train_y = np.array(train_data['Label'])
    test_x = np.array(test_data['Data'])
    test_y = np.array(test_data['Label'])
    train_num = train_x.shape[0]
    test_num = test_x.shape[0] 
    train3_index = np.where((train_y==1)|(train_y==2)|(train_y==3)) #找到前3类的标签
    test3_index = np.where((test_y==1)|(test_y==2)|(test_y==3))
    
    train3_x = train_x[train3_index[0],:]
    train3_y = train_y[train3_index[0]]
    test3_x = test_x[test3_index[0],:]
    test3_y = test_y[test3_index[0]]
    train3_num = train3_x.shape[0]
    test3_num = test3_x.shape[0]
    train_x_std = StandardScaler().fit_transform(train3_x)
    test_x_std = StandardScaler().fit_transform(test3_x) 
    figure_size = 32*32
    train_xr = StandardScaler().fit_transform(train3_x[:,:figure_size-1]);
    train_xg = StandardScaler().fit_transform(train3_x[:,figure_size:2*figure_size-1]);
    train_xb = StandardScaler().fit_transform(train3_x[:,2*figure_size:3*figure_size-1]);
    test_xr = StandardScaler().fit_transform(test3_x[:,:figure_size-1]);
    test_xg = StandardScaler().fit_transform(test3_x[:,figure_size:2*figure_size-1]);
    test_xb = StandardScaler().fit_transform(test3_x[:,2*figure_size:3*figure_size-1]);
  
    N_pca = np.linspace(10,150,15,dtype=np.int)  #PCA降维数
    score_test = np.zeros(len(N_pca))
    score_train = np.zeros(len(N_pca))
    epoch = 0
    for n_pca in N_pca:
        pca1 = PCA(n_components=n_pca,whiten=True)
        pca2 = PCA(n_components=n_pca,whiten=True)
        pca3 = PCA(n_components=n_pca,whiten=True)
        train_xr_pca = pca1.fit_transform(train_xr)
        test_xr_pca = pca1.transform(test_xr)
        train_xg_pca = pca2.fit_transform(train_xg)
        test_xg_pca = pca2.transform(test_xg)
        train_xb_pca = pca3.fit_transform(train_xb)
        test_xb_pca = pca3.transform(test_xb)

        n_classes = 3  #总类数
        n_channels = 3
        n_features = test_xr_pca.shape[1] 
        mu = np.zeros((n_channels,n_classes,n_features))
        var = np.zeros((n_channels,n_classes,n_features,n_features))
        get_mean_var(train_xr_pca,train_y,mu[0],var[0],n_classes)
        get_mean_var(train_xg_pca,train_y,mu[1],var[1],n_classes)
        get_mean_var(train_xb_pca,train_y,mu[2],var[2],n_classes)
        '''
        C = 0    
        for i in range(train3_num):
            p = Gauss_predict(train_x_pca[i,:],mu,var)
            if p+1 == test3_y[i]:
                C+=1
        score_train[epoch] = C/train3_num
        print('PCA:',n_pca,' 训练集正确率：',score_train[epoch]*100,'%')
        '''
        C = 0    
        for i in range(test3_num):
            p1 = Gauss_predict(test_xr_pca[i,:],mu[0],var[0])
            p2 = Gauss_predict(test_xg_pca[i,:],mu[1],var[1])
            p3 = Gauss_predict(test_xb_pca[i,:],mu[2],var[2])
            predict = np.zeros(n_channels,dtype = np.int)
            predict[0] = np.argmax(p1)
            predict[1] = np.argmax(p2)
            predict[2] = np.argmax(p3)
            if np.argmax(p1+p2+p3)+1 == test3_y[i]:
                C+=1
            print(p1[predict[0]],p2[predict[0]],p3[predict[0]],test3_y[i])
        score_test[epoch] = C/test3_num
        print('PCA:',n_pca,' 测试集正确率：',score_test[epoch]*100,'%')
        epoch +=1
    plt.figure(1)
    plt.plot(N_pca,score_test,label="Test Data",color="red",linewidth=2)
    plt.plot(N_pca,score_train,label="Train Data",color="blue",linewidth=2)
    plt.xlabel("PCA components")
    plt.ylabel("Accuracy%")
    plt.title("PCA components-Accuracy")
    plt.show(1)
    #用sklearn库来进行预测的结果，与自己编写的贝叶斯决策器结果比较
"""
        clf = GaussianNB().fit(train_x_pca, train3_y)
        result = clf.predict(test_x_pca)
        result2 = clf.predict(train_x_pca)    
        count = 0
        for i in range(test3_num):
            if result[i]==test3_y[i]:
                count += 1 
        score2 = count/test3_num
        print(score2)


        count2 = 0
        print(score)
        for i in range(train3_num):
            if result[i]==train3_y[i]:
                count2 += 1 
        score2 = count2/train3_num
        print(score2)
"""