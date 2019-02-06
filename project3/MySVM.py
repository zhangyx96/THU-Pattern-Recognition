# -*- coding: utf-8 -*-
"""
Created on 2018/12/13
@author: Zhang Yuanxin
"""
import numpy as np
import random as rnd
import math

class SVM():
    def __init__(self, epoch=1000, kernel_type='sigmoid', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'RBF':self.kernel_RBF,
            'exp': self.kernel_exp,
            'sigmoid': self.kernel_sigmoid
        }
        self.C = C
        self.epsilon = epsilon
        self.epoch = epoch
        self.kernel_type = kernel_type


    def fit(self, X, y):
        n = X.shape[0]
        alpha = np.zeros(n)
        kernel = self.kernels[self.kernel_type]
        for i in range(self.epoch):
            alpha_p = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                #计算L，H
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # 计算w,b
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # 计算Ei,Ej
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # 更新alpha的值
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            Diff = np.linalg.norm(alpha - alpha_p)
            if Diff < self.epsilon:
                break

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        SupportVectors = X[alpha_idx, :]
        return SupportVectors
    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))
    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # 计算E
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_j, alpha_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
        else:
            return (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))
    def get_rnd(self, a,b,c):
        i = c
        while i == c:
            i = rnd.randint(a,b)
        return i

    # Define kernels
    def kernel_linear(self, x1, x2): #线性核
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2): #平方核
        return (np.dot(x1, x2.T) ** 2)
    def kernel_RBF(self,x1,x2,gamma = 0.02):  #高斯核
        return np.exp(-np.linalg.norm(x1-x2)**2*gamma)
    def kernel_exp(self,x1,x2,gamma = 0.02):  #指数核
        return np.exp(-np.linalg.norm(x1-x2)*gamma)
    def kernel_sigmoid(self,x1,x2,alpha = 1,c = 0):  #sigmoid核
        return np.tanh(alpha*np.dot(x1, x2.T)+c)

    def SavePara(self,filename): #保存模型
        np.save(filename+'.npy',(self.w,self.b))
    def LoadPara(self,filename): #读取模型
        self.w,self.b = np.load(filename+'.npy')
