import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from collections import Counter
from random import randint
from pprint import pprint
from random import shuffle
from time import time
from util import *

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def cuda_variable (V):
    if USE_CUDA: return V.cuda()
    else: return V


embed_dim = 32
n_classes = 3
n_hidden_layer_cells = 20
n_epochs = 50

num_samples = None

def get_word_ix_tensor(sent,word_to_ix):
    word_ixs = [word_to_ix[w] for w in sent.split()]
    word_ixs = cuda_variable(Variable(torch.LongTensor(word_ixs)))

    return word_ixs

class lstm_spooky(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size,num_classes):
        super(lstm_spooky, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #layers
        # dim1: seq, dim2: minibatch size (always 1 in our case), dim3: word embedding size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2class = nn.Linear(hidden_dim, num_classes)

        #init routines invocation
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (cuda_variable(Variable(torch.zeros(1, 1, self.hidden_dim))),
                cuda_variable(Variable(torch.zeros(1, 1, self.hidden_dim))))

    def forward(self, sentence_tensor):
        embeds = self.word_embeddings(sentence_tensor)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence_tensor), 1, -1), self.hidden) #reshape 2D to 3D to set mini batch dim = 1
        raw_predictions = self.hidden2class(lstm_out.view(len(sentence_tensor), -1))
        try:
            predictions = F.log_softmax(raw_predictions,dim=1)
        except:
            predictions = F.log_softmax(raw_predictions)
        pred_class_lsoftmax = predictions[-1,:]
        return pred_class_lsoftmax

    def predict(self,sentence_tensor):
        self.init_hidden()
        embeds = self.word_embeddings(sentence_tensor)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence_tensor), 1, -1),
                                          self.hidden)  # reshape 2D to 3D to set mini batch dim = 1
        raw_predictions = self.hidden2class(lstm_out.view(len(sentence_tensor), -1))
        try:
            predictions = F.log_softmax(raw_predictions,dim=1)
        except:
            predictions = F.log_softmax(raw_predictions)
        _,max_index_tensor = predictions[-1, :].data.max(0)
        max_index = max_index_tensor[0]
        return max_index



train_data, test_data = load_train_test_data(num_samples)
train_sentences = [sent.lower() for sent in train_data['text'].tolist()]
test_sentences = [sent.lower() for sent in test_data['text'].tolist()]
y_train = [cuda_variable(Variable(torch.LongTensor([auth_id]))) for auth_id in train_data['author']]

all_sents = train_sentences + test_sentences
vocab = sorted(list(set([w for sent in all_sents for w in sent.split()])))
vocab_size = len(vocab)
print ('loaded vocab of len: ',vocab_size)

word_to_ix = {w:i for i,w in enumerate(vocab)}
ix_to_word = {v:k for k,v in word_to_ix.items()}

x_train = [get_word_ix_tensor(sent,word_to_ix) for sent in train_sentences]

x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.3)

model = lstm_spooky (embedding_dim=embed_dim,
                   hidden_dim=n_hidden_layer_cells,
                   vocab_size=vocab_size,
                   num_classes=n_classes)
if USE_CUDA: model = model.cuda()

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(n_epochs):
    t0 = time()
    losses = []
    shuffle(list(zip(x_train,y_train)))
    for x,y in list(zip(x_train,y_train)):
        model.zero_grad()
        model.hidden = model.init_hidden()
        y_hat = model(x).view(1,-1)
        loss = loss_function(y_hat,y)
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

    if epoch%1 == 0:
            print ('loss at epoch {} is {}, took {} sec.'.format(epoch,np.array(losses).mean(),round(time()-t0,2)))



y_pred = [model.predict(x) for x,y in list(zip(x_valid,y_valid))]
y_valid = [y.data[0] for y in y_valid]
print (classification_report(y_valid,y_pred))