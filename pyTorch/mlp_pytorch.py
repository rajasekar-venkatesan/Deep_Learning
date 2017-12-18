import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd, numpy as np,csv
import os,json,cPickle as pickle,sys

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from collections import Counter
from random import randint
from pprint import pprint


from util import *

class MLP(nn.Module):
    def __init__(self,input_layer_size,hidden_layer_size,output_layer_size):
        super(MLP, self).__init__()
        #affine maps
        self.hidden_layer = nn.Linear(input_layer_size,hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size,output_layer_size)
        #activation functions
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self,x):
        out_log_softmax = self.log_softmax(self.output_layer(self.relu(self.hidden_layer(x))))
        return out_log_softmax


num_samples = 100
mode = 'all'

train_data, test_data = load_train_test_data(num_samples)
train_sentences = train_data['text'].tolist()
test_sentences = test_data['text'].tolist()

vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit(train_sentences+test_sentences)

x_train = vectorizer.transform(train_sentences).todense()
x_test = vectorizer.transform(test_sentences).todense()
y_train = np.array(train_data['author'])

print 'test arrays shape: ', x_train.shape, y_train.shape
print 'label dist',Counter(y_train)

y_train = categorical_labeler(y_train)

net = MLP (input_layer_size=x_train.shape[1],
           hidden_layer_size=100,
           output_layer_size=3)

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(101):
    for x,y in zip(x_train,y_train):
        optimizer.zero_grad() #should it be net.zero_grad?
        x = torch.FloatTensor(x.tolist()[0])
        x = Variable(x)
        y_hat = net(x).view(1,-1)
        y = Variable(torch.LongTensor(y))#.view(1,-1)
        loss = loss_function(y_hat,y)

        loss.backward()
        optimizer.step()

    if epoch%10 == 0:
            print 'loss at epoch {} is {}'.format(epoch,loss.data[0])
















