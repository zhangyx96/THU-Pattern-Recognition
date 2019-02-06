# -*- coding: utf-8 -*-
"""
Created on 2018/12/13
@author: Zhang Yuanxin
"""
import numpy as np
import os,sys
import scipy.io as sio
import logging
import time
from skimage import feature,color

from MySVM import SVM
from itertools import combinations

def LoadData(data_path):
    DataList = ['labels_name3.mat','train_data3.mat','test_data3.mat']
    #labels_name = sio.loadmat(data_path+data_list[0])
    TrainData = sio.loadmat(data_path+ DataList[1])
    TestData = sio.loadmat(data_path+ DataList[2])
    TrainX = np.array(TrainData['Data'])
    TrainY = np.array(TrainData['Label'])
    TestX = np.array(TestData['Data'])
    TestY = np.array(TestData['Label'])
    return TrainX, TrainY, TestX, TestY

def DataGroup(X,y,ClassNum,labels):
    Group = []
    GpNum = []
    for i in range(ClassNum):
        index = np.where(y == labels[i])[0]
        Group.append(X[index,:])
        GpNum.append(len(index))
    return Group,GpNum


def Hog(X,Filename,New = False):
    if os.path.isfile(Filename+'.npy') and not New:
        HogFeature = np.load(Filename+'.npy')
    else:
        X = np.reshape(X,(X.shape[0],3,-1))
        X = np.transpose(X,(0,2,1))
        X = color.rgb2gray(X)
        X = np.reshape(X,(X.shape[0],32,32))
        HogFeature = []
        for i in range(X.shape[0]):
            hf = feature.hog(X[i], orientations=12,
                             pixels_per_cell=(6, 6), cells_per_block=(3, 3))
            HogFeature.append(hf)
        np.save(Filename+'.npy',HogFeature)
    return np.array(HogFeature)

def Accuracy(y1,y2):
    assert len(y1)==len(y2)
    return np.sum(y1==y2)/len(y1)

def ModelTrain(comb,X,Num):
    Model = SVM(kernel_type='sigmoid')
    x = np.vstack((X[comb[0]],X[comb[1]]))
    y = np.ones(Num[comb[0]]+Num[comb[1]])
    y[0:Num[comb[0]]] = -1
    SupportVector= Model.fit(x, y)
    return Model

def ModelEvaluate(comb,X,Num,M):
    x = np.vstack((X[comb[0]],X[comb[1]]))
    y = np.ones(Num[comb[0]]+Num[comb[1]])
    y[0:Num[comb[0]]] = -1
    Pred = M.predict(x)
    Score = Accuracy(Pred,y)
    return Score

def Vote(ClassNum,ModelNum,Model,comb,X,labels):
    Scores = np.zeros((X.shape[0],ClassNum))
    for i in range(ModelNum):
        Pred = Model[i].predict(X)
        for j in range(Pred.shape[0]):
            if Pred[j] == -1:
                Scores[j, comb[i][0]] += 1
            else:
                Scores[j, comb[i][1]] += 1
    Result = np.zeros(X.shape[0])
    for k in range(X.shape[0]):
        Result[k] = np.argmax(Scores[k,:])
        Result[k] = labels[int(Result[k])]
    return np.array(Result)

def SetLog():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    LogPath = os.path.dirname(os.getcwd()) + '/project3/Logs/'
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



if __name__ == "__main__":
    kwargs = {}
    data_path = 'D:/Office/研究生课程/模式识别/project3/data3/'
    logger = SetLog()
    TrainX, TrainY, TestX, TestY = LoadData(data_path)
    New = True
    TrainHogX = Hog(TrainX,'TrainHogX',New)
    TestHogX = Hog(TestX,'TestHogX', New)
    logger.info('Successfully extract the Hog feature')
    logger.info('FeatureLength:'+str(TrainHogX.shape[1]))
    ClassNum = 5
    labels = [0, 6, 7, 8, 9]
    assert len(labels)==ClassNum
    TrainGp,GpNum = DataGroup(TrainHogX,TrainY,ClassNum,labels)
    TestGp,GpNum2 = DataGroup(TestHogX,TestY,ClassNum,labels)

    ModelNum = int(ClassNum*(ClassNum-1)/2)
    Comb = list(combinations(range(ClassNum),2))
    if os.path.isfile('Model.npy'):
        Model = np.load('Model.npy')
        logger.info("Successfully Load the SVM Model")
    else:
        Model = []
        for c in Comb:
            M = ModelTrain(c,TrainGp,GpNum)
            ModelScore = ModelEvaluate(c,TestGp,GpNum2,M)
            logger.info(str(c)+','+str(ModelScore))
            Model.append(M)
        np.save('Model.npy',Model)

    Result = Vote(ClassNum,ModelNum,Model,Comb,TestHogX,labels)
    Score = Accuracy(np.reshape(Result,(-1,1)),TestY)
    logger.info('Score:'+str(Score))
















