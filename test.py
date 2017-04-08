#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:37:07 2017

@author: sachin
"""

import torch
import vocab

from collections import Counter
c = Counter(['hello', 'world'])
v = vocab.Vocab(c, wv_type='glove.840B')
print(v.itos)
print(v.vectors[v.stoi["hello"]])
