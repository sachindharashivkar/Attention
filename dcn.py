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
from nltk.tokenize import word_tokenize
from collections import Counter
import json
import vocab
import os


def build_vocab(): 
    dataset_file = "train-v1.1.json"
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
    token_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i]["paragraphs"])):
            passage = dataset[i]["paragraphs"][j]["context"]
            passage = passage.replace("''", '" ')
            passage = passage.replace("``", '" ')
            token_list.extend(word_tokenize(passage))
            for k in range(len(dataset[i]["paragraphs"][j]["qas"])):
                question = dataset[i]["paragraphs"][j]["qas"][k]["question"]
                token_list.extend(word_tokenize(question))
    c = Counter(token_list)
    v = vocab.Vocab(c, wv_type='glove.840B')
    del c
    del token_list
    return v

def open_files():
    context_file = open(os.getcwd() + "/train.context", 'r')
    question_file = open(os.getcwd() + "/train.question","r")
    answer_span_file = open(os.getcwd() + "/train.span","r")
    return context_file, question_file, answer_span_file

def to_scalar(v):
    if isinstance(v, Variable):
        return v.data.view(-1).tolist()[0]
    else:
        return v.view(-1).tolist()[0]

class document_encoder(nn.Module):
    def __init__(self):
        super(document_encoder, self).__init__()
        self.fc   = nn.Linear(200, 200)
        self.lstm = nn.LSTM(300, 200)
        self.hidden = None
        
    def forward(self, x, q):
        self.hidden = None
        D, self.hidden = self.lstm(x, self.hidden)
        if q =="Q" :
            return F.relu(self.fc(D.squeeze()))   
        return D.squeeze()

class coattention_encoder(nn.Module):
    def __init__(self):
        super(coattention_encoder, self).__init__()
        self.blstm = nn.LSTM(600,200,bidirectional = True)
        self.hid = None

    def forward(self, D, Q):
        self.hid = None
        L = torch.mm(Q,D.t())
        Aq = F.softmax(L)
        Ad = F.softmax(L.t())
        Cq = torch.mm(Aq,D)
        Cd = torch.mm(Ad,torch.cat((Q,Cq),1))
        U , self.hid = self.blstm(torch.cat((D,Cd),1).unsqueeze(1),self.hid)
        return U.squeeze()[:-1,:]


class HMN(nn.Module):
    def __init__(self):
        super(HMN, self).__init__()
        self.fc1 = nn.Linear(1000, 200, bias =  False)
        self.fc2 = nn.Linear(600, 3200)
        self.fc3 = nn.Linear(200, 3200)
        self.fc4 = nn.Linear(400, 16) 

    def forward(self, U, h, s, e):
        r = F.tanh(self.fc1(torch.cat((h.squeeze(),s.squeeze(),e.squeeze()),0).unsqueeze(0)))
        R = []
        for i in range(U.size()[0]):
            R.append(r)
        r1 = torch.stack(R).squeeze()
        m1 = torch.max(((self.fc2(torch.cat((U, r1),1))).view(U.size()[0],16,-1)),1)[0].squeeze()
        m2 = torch.max(((self.fc3(m1)).view(U.size()[0],16,-1)),1)[0].squeeze()
        alpha = torch.max(((self.fc4(torch.cat((m1, m2),1))).view(U.size()[0],16,-1)),1)[0].squeeze()
        return alpha

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.start_hmn = HMN()
        self.end_hmn = HMN()
        self.lstm = nn.LSTM(800, 200)
        self.hid = None
        self.start = 0
        self.end = 0
        self.past_start = 0
        self.past_end = 0
        self.iterations = 4
        
    def forward(self, U):
        self.hid = None
        self.start = 0
        self.end = 0
        starts = []
        ends = []
        out , self.hid = self.lstm(torch.cat((U[self.start],U[self.end])).view(1,1,-1), self.hid)
        alpha = self.start_hmn(U, out, U[self.start].view(1,-1), U[self.end].view(1,-1))
        beta = self.end_hmn(U, out, U[self.start].view(1,-1), U[self.end].view(1,-1))
        _, self.start = alpha.max(0)
        _, self.end = beta.max(0)
        start = to_scalar(self.start)
        end = to_scalar(self.end)
        starts.append(alpha.view(1,-1))
        ends.append(beta.view(1,-1))
        for n in range(3):
            out , self.hid = self.lstm(torch.cat((U[self.start.data],U[self.end.data])).view(1,1,-1), self.hid)
            alpha = self.start_hmn(U, out, U[self.start.data].view(1,-1), U[self.end.data].view(1,-1))
            beta = self.end_hmn(U, out, U[self.start.data].view(1,-1), U[self.end.data].view(1,-1))
            _, self.start = alpha.max(0)
            _, self.end = beta.max(0)
            starts.append(alpha.view(1,-1))
            ends.append(beta.view(1,-1))
            if start == to_scalar(self.start) and end == to_scalar(self.end):
                break
            start = to_scalar(self.start)
            end = to_scalar(self.end)
        return starts, ends
    
class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()
        self.doc_encoder = document_encoder()
        self.co_encoder = coattention_encoder()
        self.decode = decoder()
        
    def forward(self, x , y):
        D = self.doc_encoder(x, "D")  
        Q = self.doc_encoder(y, "Q")
        U = self.co_encoder(D, Q)
        return self.decode(U)
    
def main():            
    v = build_vocab()
    model = DCN().cuda()
    optimizer = optim.Adam(model.parameters())
    #model = torch.load(os.getcwd() + "/mod")
    #optimizer = torch.load(os.getcwd() + "/opt")
    end_vectors =nn.Embedding(2,300) 
    num_epoch = 10
    i = 0
    for epoch in range(num_epoch):
        total_loss = 0
        print ("epoch:", epoch)
        c, q1, s = open_files()
        for line, question, span in zip(c, q1, s):
            passage = word_tokenize(line)
            que = word_tokenize(question)
            doc_vector = []
            question_vector = []
            for k in passage:
                doc_vector.append(Variable(v.vectors[v.stoi[k]], requires_grad = False))
            for k in que:
                question_vector.append(Variable(v.vectors[v.stoi[k]] , requires_grad =  False))            
            doc_vector.append(torch.squeeze(end_vectors(Variable(torch.LongTensor([0])))))  
            question_vector.append(torch.squeeze(end_vectors(Variable(torch.LongTensor([1])))))            
            d = torch.cat(doc_vector).view(len(doc_vector), 1, -1)
            q = torch.cat(question_vector).view(len(question_vector), 1, -1)
            sp = word_tokenize(span)
            start , end = model(d.cuda(), q.cuda())
            as1 = torch.LongTensor(1,len(start)).cuda()
            as1[0] = int(sp[0])
            ae1 = torch.LongTensor(1,len(start)).cuda()
            ae1[0] = int(sp[1])
            loss = F.cross_entropy(torch.cat(start),Variable(as1.squeeze())).cuda()
            loss1 = F.cross_entropy(torch.cat(end),Variable(ae1.squeeze())).cuda()
            (loss + loss1).backward()
            total_loss += to_scalar((loss+ loss1).data)
            i = i + 1
            if i % 100 == 0:
                optimizer.step()
                optimizer.zero_grad()
            if i%1000 == 0:
                print (total_loss/1000)
                i =0
                total_loss = 0
        c.close()
        q1.close()
        s.close()
        torch.save(model, os.getcwd() + "/mod")
        torch.save(optimizer, os.getcwd() + "/opt")
        torch.save(end_vectors, os.getcwd() + "/end_vectors")

if __name__ == "__main__":
    main()

# Add dropout
# Look for more gpu
#Model save done. Model  reload still remains
# error in pickle of v.stoi
