#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import numpy as np
import math
from utils import get_feature_vectors,get_label
from classifier import BinaryClassifier

class NaiveBayes(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.dimensions = args.f_dim
        self.vocab_size = args.vocab_size
        self.num_iter = args.num_iter
        self.lr = args.lr
        self.bin_feats = args.bin_feats
        self.size_cp = 0
        self.size_cn = 0 
        self.cp = 0 # probability of c+
        self.cn = 0 # probability of c-
        self.total_freq_pos = np.zeros(self.dimensions) #probability of d given c- 
        self.total_freq_neg = np.zeros(self.dimensions) #probability of d given c- 
        self.final_list = np.zeros(self.dimensions) # number of d given c+
        #raise NotImplementedError
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        #calculate cp which is c+ 
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        np.random.seed(5) #this line is to ensure that you get the same shuffled order everytime
        np.random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        
        feature = get_feature_vectors(train_data[0],self.bin_feats)
        i = 0 
        for x in feature:
            if train_data[1][i] == 1:
                self.total_freq_pos = np.add(self.total_freq_pos,x) #probability of d given c- 
                self.size_cp = self.size_cp + 1
            elif train_data[1][i] == -1:
                self.total_freq_neg = np.add(self.total_freq_neg,x) #probability of d given c-  
                self.size_cn = self.size_cn + 1
            i = i + 1
        wordcountpos = 0
        wordcountneg = 0
        for m in self.total_freq_pos:
            wordcountpos = wordcountpos + m
        for n in self.total_freq_neg:
            wordcountneg = wordcountneg + n
        
        self.cp = self.size_cp / (self.size_cn + self.size_cp)
        self.cn = self.size_cn / (self.size_cn + self.size_cp)
        i = 0
        for p in self.total_freq_pos:
            self.total_freq_pos[i] =  ((self.total_freq_pos[i] + 1)/(wordcountpos + self.vocab_size))
            i = i + 1        
        i = 0 
        for m in self.total_freq_neg:
            self.total_freq_neg[i] = ((self.total_freq_neg[i] + 1)/(wordcountneg + self.vocab_size))
            i = i + 1
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        #@478
        feature = get_feature_vectors(test_x,self.bin_feats)
        j = 0
        for x in feature:
            finalpos = 0
            finalneg = 0
            i = 0
            for y in self.total_freq_pos:
                finalpos +=  math.log10(math.pow(y,x[i]))
                i = i + 1
            i = 0 
            for z in self.total_freq_neg:
                finalneg += math.log10(math.pow(z,x[i]))    
                i = i + 1
            finalpos += math.log10(self.size_cp)
            finalneg += math.log10(self.size_cn)
            if(finalpos > finalneg):
                self.final_list[j] = 1
            else:
                self.final_list[j] = -1
            j = j + 1
        return self.final_list
        

