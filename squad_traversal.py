#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:38:09 2017

@author: sachin
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import autograd

from nltk.tokenize import word_tokenize
from collections import Counter
import json
import vocab

dataset_file = "train-v1.1.json"
with open(dataset_file) as dataset_file:
    dataset_json = json.load(dataset_file)
dataset = dataset_json['data']

token_list = []
np = 0
nq = 0

for i in range(len(dataset)):
    for j in range(len(dataset[i]["paragraphs"])):
        passage = dataset[i]["paragraphs"][j]["context"]
        np = np + 1
        token_list.extend(word_tokenize(passage.lower()))
        for k in range(len(dataset[i]["paragraphs"][j]["qas"])):
            question = dataset[i]["paragraphs"][j]["qas"][k]["question"]
            nq = nq + 1
            token_list.extend(word_tokenize(question.lower()))
            for l in range(len(dataset[i]["paragraphs"][j]["qas"][k]["answers"])):
                answer = dataset[i]["paragraphs"][j]["qas"][k]["answers"][l]["text"]
                ans_start  = dataset[i]["paragraphs"][j]["qas"][k]["answers"][l]["answer_start"]
            
c = Counter(token_list)
v = vocab.Vocab(c, wv_type='glove.840B')
print(v.vectors[v.stoi["hello"]])

del c
del token_list

class question_encoder(nn.Module):
    def __init__(self):
        super(question_encoder, self).__init__()
        self.fc   = nn.Linear(200, 200)

    def forward(self, x):
        return F.relu(self.fc(x))     
    
class coattention_encoder(nn.Module):
    def __init__(self):
        super(coattention_encoder, self).__init__()
        self.blstm = nn.LSTM(600,200,bidirectional = True)
        self.hid = None

    def forward(self, D, Q):
        L = torch.mm(Q,D.t())
        Aq = F.softmax(L)
        Ad = F.softmax(L.t())
        Cq = torch.mm(Aq,D)
        temp = torch.cat((Q,Cq),1)
        Cd = torch.mm(Ad,temp)
        temp1 = torch.cat((D,Cd),1)
        self.hid = None
        U , self.hid = self.blstm(temp1.unsqueeze(1),self.hid)
        return U
"""   
class HMN(nn.Module):
    def __init__(self):
        super(HMN, self).__init__()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
        self.fc4 = nn.Linear()

    def forward(self, x):
        return F.relu(self.fc(x))   
        
qe = question_encoder()
ce = coattention_encoder()
end_vectors =nn.Embedding(2,300) 
for i in range(1):
    for j in range(1):
        passage = dataset[i]["paragraphs"][j]["context"]
        passage = word_tokenize(passage.lower())
        doc_vector = []
        for k in passage:
            doc_vector.append(Variable(v.vectors[v.stoi[k]]))
        doc_vector.append(torch.squeeze(end_vectors(Variable(torch.LongTensor([0])))))
        inputs = torch.cat(doc_vector).view(len(doc_vector), 1, -1)
        hidden = None
        lstm = nn.LSTM(300, 200)
        D1, hidden = lstm(inputs, hidden)   
        D = D1.squeeze()
        for k in range(len(dataset[i]["paragraphs"][j]["qas"])):
            question = dataset[i]["paragraphs"][j]["qas"][k]["question"]
            question = word_tokenize(question.lower())
            question_vector = []
            for l in question:
                question_vector.append(Variable(v.vectors[v.stoi[l]]))
            question_vector.append(torch.squeeze(end_vectors(Variable(torch.LongTensor([1])))))
            hidden = None
            inputs = torch.cat(question_vector).view(len(question_vector), 1, -1)
            Q1, hidden = lstm(inputs, hidden)   
            Q2 = Q1.squeeze()
            Q = qe(Q2)
            U = ce(D, Q)
"""