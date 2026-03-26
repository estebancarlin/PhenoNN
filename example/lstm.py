#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:42:43 2023

@author: gliu
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, target_size, input_size, hidden_size, num_layers, dropout_rate = 0.0, device = 'cpu'):
        super(LSTM, self).__init__()
        self.target_size = target_size 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                          num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size, 128) # fully connected 1  
        self.fc = nn.Linear(128, target_size) # fully connected last layer
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(dropout_rate)
        self.device=device
    
    def forward(self,x):
        outputs = []
        hn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        cn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)#internal state        
        for time_step in x.split(1, dim = 1):
            time_step = self.dropout(time_step)
            output, (hn, cn) = self.lstm(time_step, (hn, cn)) #lstm with input, hidden, and internal state
            out = output.reshape(-1, self.hidden_size)
            out = self.relu(out)
            out = self.fc_1(out) #first Dense
            out = self.relu(out) 
            out = self.fc(out) #Final Output
            outputs.append(out)
        outputs = torch.cat(outputs, dim = 1)
      
        return outputs
