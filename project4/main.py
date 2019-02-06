# -*- coding: utf-8 -*-
"""
Created on 2019/1/6
@author: Zhang Yuanxin
"""
import numpy as np
import os,sys
import scipy.io as sio
import logging
import time

from itertools import combinations
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
import sklearn.model_selection as sk_model_selection


def LoadData(data_path):
    DataList = ['train_data.mat','test_data.mat']
    TrainData = sio.loadmat(data_path+ DataList[0])
    TestData = sio.loadmat(data_path+ DataList[1])
    TrainX = np.array(TrainData['data'])
    TrainY = np.array(TrainData['label'])
    TestX = np.array(TestData['data'])
    return TrainX, TrainY, TestX

def DataGroup(X,y,ClassNum,labels):
    Group = []
    GpNum = []
    for i in range(ClassNum):
        index = np.where(y == labels[i])[0]
        Group.append(X[index,:])
        GpNum.append(len(index))
    return Group,GpNum

def SetLog():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    LogPath = os.path.dirname(os.getcwd()) + '/project4/Logs/'
    if not os.path.exists(LogPath): os.mkdir(LogPath)
    LogName = LogPath+rq+'.log'

    fh = logging.FileHandler(LogName,mode='w')
    fh.setLevel(logging.INFO)
    ch  =logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def KNN(X,Y,test_x,run_only = False):
    model = KNeighborsClassifier(n_neighbors=5)
    accs = 0
    if not run_only:
        my_cv = RepeatedKFold(n_splits = 5 ,n_repeats=1000)
        accs=sk_model_selection.cross_val_score(model, X, y=Y,cv=my_cv)
    model.fit(X,Y)
    pred = model.predict(test_x)
    return accs,pred

def MLP(X,Y,test_x,run_only = False):
    model = MLPClassifier(hidden_layer_sizes=(100,100,100))
    accs = 0
    if not run_only:
        my_cv = RepeatedKFold(n_splits = 5 ,n_repeats=1000)
        accs=sk_model_selection.cross_val_score(model, X, y=Y,cv=my_cv)
    model.fit(X,Y)
    pred = model.predict(test_x)
    return accs,pred

def SVM(X,Y,test_x,run_only = False):
    model = SVC(kernel='rbf',gamma=0.015,decision_function_shape='ovo',C = 5)
    accs = 0
    if not run_only:
        my_cv = RepeatedKFold(n_splits = 5 ,n_repeats=1000)
        accs=sk_model_selection.cross_val_score(model, X, y=Y,cv=my_cv)
    model.fit(X,Y)
    pred = model.predict(test_x)
    return accs,pred

def Bayes(X,Y,test_x,run_only = False):
    model = GaussianNB()
    accs = 0
    if not run_only:
        my_cv = RepeatedKFold(n_splits = 5 ,n_repeats=1000)
        accs=sk_model_selection.cross_val_score(model, abs(X), y=Y,cv=my_cv)
    model.fit(X,Y)
    pred = model.predict(test_x)
    return accs,pred

def SimRatio(x,y):
    assert len(x) == len(y)
    return np.sum(x==y)/len(x)

def vote(x1,x2,x3):
    N = len(x1)
    assert N==len(x2)
    assert N==len(x3)
    #assert N==len(x4)
    C = np.zeros((N,8))
    for i in range(N):
        C[i,x1[i]-1] += 1
        C[i,x2[i]-1] += 1
        C[i,x3[i]-1] += 1
        #C[i,x4[i]-1] += 0
    Pred = np.zeros(N,dtype = int)
    for i in range(N):
        max = 0
        val = 8
        for j in reversed(range(8)):
            if C[i,j] > max:
                max = C[i,j]
                val = j+1
        if max == 1:
            val = x1[i]
        Pred[i] = int(val) 
    return Pred

if __name__ == "__main__":
    data_path = 'D:/Office/研究生课程/模式识别/project4/data4/'
    logger = SetLog()
    TrainX, TrainY, TestX = LoadData(data_path)
    ClassNum = 8
    labels = [1, 2, 3, 4, 5, 6,7,8]
    assert len(labels)==ClassNum
    sm = SMOTE(random_state=42)
    TrainX_Res, TrainY_Res = sm.fit_resample(TrainX, TrainY)

    #Score1,Pred1 = KNN(TrainX_Res,TrainY_Res,TestX)
    #Score2,Pred2 = MLP(TrainX_Res,TrainY_Res,TestX)
    #Score3,Pred3 = SVM(TrainX_Res,TrainY_Res,TestX)
    #Score4,Pred4 = Bayes(TrainX_Res,TrainY_Res,TestX)

    Score1,Pred1 = KNN(TrainX_Res,TrainY_Res,TestX,run_only=True)
    Score2,Pred2 = MLP(TrainX_Res,TrainY_Res,TestX,run_only=True)
    Score3,Pred3 = SVM(TrainX_Res,TrainY_Res,TestX,run_only=True)

    #logger.info(np.mean(Score1))
    #logger.info(np.mean(Score2))
    #logger.info(np.mean(Score3))
    #logger.info(np.mean(Score4))

    mat_path = 'D:/Office/研究生课程/模式识别/project4/'
    
    sio.savemat(mat_path+'KNN.mat', {'lable': Pred1})
    sio.savemat(mat_path+'MLP.mat', {'lable': Pred2})
    sio.savemat(mat_path+'SVM.mat', {'lable': Pred3})
    #sio.savemat(mat_path+'Bayes.mat', {'lable': Pred4})
    
    S12 = SimRatio(Pred1,Pred2)
    S13 = SimRatio(Pred1,Pred3)
    S23 = SimRatio(Pred2,Pred3)

    print(S12,S13,S23)
    
    Pred = vote(Pred1,Pred2,Pred3)
    sio.savemat(mat_path+'Voted.mat', {'lable': Pred})

