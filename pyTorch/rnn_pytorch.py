from __future__ import print_function

import torch, torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import accuracy_score, classification_report

from util import *

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def cuda_variable (V,requires_grad=False):
    V = Variable(V, requires_grad)
    if USE_CUDA: return V.cuda()
    else: return V


num_samples = 100

embedding_dim = 16
hidden_size = 30
num_classes = 3
lr = 0.1
epochs = 10

def make_y_onehot(y):
    one_hot_y = [1 if i == y else 0 for i in xrange(num_classes)]
    one_hot_y = cuda_variable(FloatTensor(one_hot_y).view(1,-1))
    return one_hot_y

def get_word_ix_tensor(sent,word_to_ix):
    word_ixs = [word_to_ix[w] for w in sent.split()]
    word_ixs = cuda_variable(LongTensor(word_ixs))
    return word_ixs

class RNN():
    def __init__(self,vocab_size,ip_size,hidden_size,op_size):
        self.hidden_size = hidden_size
        self.op_size = op_size
        if USE_CUDA:
            self.word_vectors = nn.Embedding(vocab_size, ip_size).cuda()
        else:
            self.word_vectors = nn.Embedding(vocab_size, ip_size)
        self.i2h = cuda_variable(torch.randn(ip_size+hidden_size,hidden_size),requires_grad = True)
        self.h2o = cuda_variable(torch.randn(hidden_size,op_size),requires_grad = True)

    def init_hidden (self):
        return cuda_variable(torch.zeros(1,self.hidden_size))

    def forward(self,word_ids):
        word_vecs = self.word_vectors(word_ids)

        hidden = self.init_hidden()
        for x in word_vecs:
            x = torch.cat((x.view(1,-1),hidden),dim=1)
            hidden = F.tanh(torch.mm(x,self.i2h))
            out = F.softmax(torch.mm(hidden,self.h2o))
        return out

    def compute_loss(self,y,y_hat):
        loss = (y-y_hat)**2
        loss = torch.sum(loss)
        return loss

    def backward(self,loss,lr):
        loss.backward()
        self.i2h.data -= lr * self.i2h.grad.data
        self.h2o.data -= lr * self.h2o.grad.data

        self.i2h.grad.data = torch.zeros(self.i2h.size())
        self.i2h.grad.data = torch.zeros(self.i2h.size())

    def predict(self,word_ids):
        word_vecs = self.word_vectors(word_ids)

        hidden = self.init_hidden()
        for x in word_vecs:
            x = torch.cat((x.view(1, -1), hidden), dim=1)
            hidden = F.tanh(torch.mm(x, self.i2h))
            out = F.softmax(torch.mm(hidden, self.h2o))

        top_prob,top_i = torch.topk(out,1)
        return top_i.data[0][0]




train_data, test_data = load_train_test_data(num_samples)
train_sentences = [sent.lower() for sent in train_data['text'].tolist()]
test_sentences = [sent.lower() for sent in test_data['text'].tolist()]
y_train = train_data['author'].tolist()

all_sents = train_sentences + test_sentences
vocab = sorted(list(set([w for sent in all_sents for w in sent.split()])))
vocab_size = len(vocab)
print ('loaded vocab of len: ',vocab_size)

word_to_ix = {w:i for i,w in enumerate(vocab)}
ix_to_word = {v:k for k,v in word_to_ix.items()}

x_train = [get_word_ix_tensor(sent,word_to_ix) for sent in train_sentences]

x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.3)
y_train = [make_y_onehot(auth_id) for auth_id in y_train] #list of variables

rnn = RNN(vocab_size=vocab_size,ip_size=embedding_dim,hidden_size=hidden_size,op_size=num_classes)

for e in range(1,epochs+1):
    losses = []
    t0 = time()
    for x,y in list(zip(x_train,y_train)):
        y_pred = rnn.forward(x)
        loss = rnn.compute_loss(y,y_pred)
        rnn.backward(loss,lr)
        losses.append(loss.data[0])

    print ('epoch: {}, time: {}, loss: {}'.format(e,round(time()-t0),round(sum(losses)/len(losses),6)))


y_pred = [rnn.predict(x) for x in x_valid]
print (classification_report(y_valid,y_pred))
print ('accuracy',accuracy_score(y_valid,y_pred))