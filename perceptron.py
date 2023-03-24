#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import numpy as np

from classifier import BinaryClassifier
from utils import get_feature_vectors,get_label
class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.dimensions = args.f_dim
        self.vocab_size = args.vocab_size
        self.num_iter = args.num_iter
        self.lr = args.lr
        self.bin_feats = args.bin_feats
        self.w = np.zeros(self.dimensions)
        self.b = 0
        #raise NotImplementedError
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        np.random.seed(5) #this line is to ensure that you get the same shuffled order everytime
        np.random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        feature = get_feature_vectors(train_data[0],self.bin_feats)
        for i in range(self.num_iter):
            for x,y in zip(feature,train_data[1]):
                y1 = np.dot(self.w,x) + self.b
                if y1 > 0:
                    y1 = 1
                else:
                    y1 t
exit -1
                if y != y1:
                    self.w = self.w + (self.lr * y * np.array(x)) 
                    self.b = self.b + (self.lr *y1)
        #raise NotImplementedError
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        ret = []
        feature2 = get_feature_vectors(test_x,self.bin_feats)
        #for i in range(self.num_iter):
        for x in feature2:
            y1 = np.dot(self.w,x) + self.b
            if y1 > 0:
                ret.append(1)
            else:
                ret.append(-1)
        return ret
        #raise NotImplementedError


class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.dimensions = args.f_dim
        self.vocab_size = args.vocab_size
        self.num_iter = args.num_iter
        self.lr = args.lr
        self.bin_feats = args.bin_feats
        self.w = np.zeros(self.dimensions)
        self.b = 0
        self.sur = 1
        #raise NotImplementedError
                
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        np.random.seed(5) #this line is to ensure that you get the same shuffled order everytime
        np.random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        feature = get_feature_vectors(train_data[0],self.bin_feats)
        for i in range(self.num_iter):
            for x,y in zip(feature,train_data[1]):
                y1 = np.dot(self.w,x) + self.b
                if y1 > 0:
                    y1 = 1
                else:
                    y1 = -1
                if y != y1:
                    w1 = self.w + (self.lr * y * np.array(x)) 
                    self.w = ((self.sur * self.w) + w1)/(self.sur + 1)
                    self.b = self.b + (self.lr *y1)
                    self.sur = 1
                else:
                    self.sur = self.sur + 1
        #raise NotImplementedError
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        ret = []
        feature2 = get_feature_vectors(test_x,self.bin_feats)
        for i in range(self.num_iter):
            for x in feature2:
                y1 = np.dot(self.w,x) + self.b
                if y1 > 0:
                    ret.append(1)
                else:
                    ret.append(-1)
        return ret
        #raise NotImplementedError

