import numpy as np
import scipy.io as sio 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def Gauss_predict(x,mu,var):
    
    px = np.zeros(len(mu))
    for i in range(len(mu)):    
        px[i] = -0.5*np.log(np.linalg.det(var[i,:,:]))
        px[i] += -0.5*((x-mu[i]).reshape(1,-1)).dot(np.linalg.inv(var[i,:,:])).dot((x-mu[i]).reshape(-1,1))
    return np.argmax(px)

def _Gauss_predict(x,mu,sigma):
    joint_log_likelihood = np.zeros(len(mu))
    for i in range(len(mu)):
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * sigma[i, :]))
        n_ij -= 0.5 * np.sum(((x - mu[i]) ** 2) /(sigma[i, :]))
        joint_log_likelihood[i] = n_ij
    return np.argmax(joint_log_likelihood)




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
    train3_index = np.where((train_y==1)|(train_y==2)|(train_y==3))
    test3_index = np.where((test_y==1)|(test_y==2)|(test_y==3))
    
    train3_x = train_x[train3_index[0],:]
    train3_y = train_y[train3_index[0]]
    test3_x = test_x[test3_index[0],:]
    test3_y = test_y[test3_index[0]]
    train3_num = train3_x.shape[0]
    test3_num = test3_x.shape[0] 
    train_x_std = StandardScaler().fit_transform(train3_x)  #0均值化处理
    test_x_std = StandardScaler().fit_transform(test3_x) 
  
    n_pca = 20  #PCA降维数
    pca = PCA(n_components=n_pca,whiten=True)
    train_x_pca = pca.fit_transform(train_x_std)
    test_x_pca = pca.transform(test_x_std)

    n_classes = 3  #总类数
    n_features = test_x_pca.shape[1] 
    mu = np.zeros((n_classes,n_features))
    var = np.zeros((n_classes,n_features,n_features))
    #_var = np.zeros((n_classes,n_features))

    for i in range(3):
        l_index = np.where(train_y==i+1)[0]
        X_i = train_x_pca[l_index,:]
        M = np.average(X_i,axis=0)
        V = np.zeros((n_features,n_features)) 
        for j in range(X_i.shape[0]):
            Temp = X_i[j,:]-M
            #V += np.dot(Temp.reshape(-1,1), Temp.reshape(1,-1))
            V += np.dot(Temp.reshape(-1,1), Temp.reshape(1,-1))         
        mu[i,:]= M    
        #var[i,:,:] = V/(X_i.shape[0])
        var[i,:,:] = np.cov(X_i.T)
        #_var[i,:] = np.average((X_i-M)**2,axis=0)

    C = 0    
    for i in range(test3_num):
        p = Gauss_predict(test_x_pca[i,:],mu,var)
        if p+1 == test3_y[i]:
            C+=1
    score = C/test3_num
    print(score)



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