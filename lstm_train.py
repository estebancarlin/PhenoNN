#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# build the LSTM models based on dataloader 
LSTM model: M0, Mfull, Mn
predictor: clim + static 
PFT:DB, EN, GR
@author: gliu
"""
import os
import dataloader_phenodata as dl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from lstm import LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('device set to {}'.format(device))
         

def train_model(
        train_loader, 
        valid_loader, 
        model, 
        loss_function, 
        optimizer, 
        scheduler,
        epochs,
        patience,
        nr_features,
        target = 'gcc'):
    
    patience_counter = 0
    best_loss = 10e10
    train_loss_epochs = []
    valid_loss_epochs = [] 
    
    tepoch = tqdm(range(epochs))
    for epoch in tepoch: 
        model.train()
        train_total_losses = 0    
        
        for batch_number, batch in enumerate(train_loader, 1):        
            optimizer.zero_grad() 
            if nr_features == 6:
                X_train = batch['features'][:,:,:6]
            elif nr_features == 8:
                X_train = torch.cat((batch['features'][:,:,:6],batch['features'][:,:,8:10]),axis=-1)
            elif nr_features == 9:
                X_train = torch.cat((batch['features'][:,:,:7],batch['features'][:,:,8:10]),axis=-1)
            else:
                X_train = torch.cat((batch['features'][:,:,:7],batch['features'][:,:,8:]),axis=-1)
            
            if target == 'gcc':
                y_train = batch['target'][:,:,0] 
            elif target == 'rcc':
                y_train = batch['target'][:,:,1] 
            elif target == 'gcc_lowess':
                y_train = batch['target'][:,:,2]     
            elif target == 'rcc_lowess':
                y_train = batch['target'][:,:,3]
            elif target == 'gcc_norm':
                y_train = batch['target'][:,:,4] 
            elif target == 'rcc_norm':
                y_train = batch['target'][:,:,5] 
            elif target == 'gcc_lowess_norm':
                y_train = batch['target'][:,:,6]     
            else:
                y_train = batch['target'][:,:,7]  
            
            outputs = model(X_train) 
            outputs = outputs[:,365:365 + y_train.shape[-1]]
            loss = loss_function(outputs, y_train)             
                      
            loss.backward()            
            optimizer.step() 

            train_total_losses += loss.item()
        
        train_loss = train_total_losses / len(train_loader)                   
        valid_loss = valid_model(valid_loader, model, loss_function, nr_features, target)
        train_loss_epochs.append(train_loss)
        valid_loss_epochs.append(valid_loss)
        
        scheduler.step()
        
        # Early stopping
        tepoch.set_postfix(train_loss = train_loss, validation_loss = valid_loss) 
        if valid_loss > best_loss:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping!\nStart to test process.')
                print(epoch)
                return model, train_loss_epochs, valid_loss_epochs, epoch         
        else:
            patience_counter = 0
            best_loss = valid_loss  
      
    return model, train_loss_epochs, valid_loss_epochs, epoch    

    
def valid_model(valid_loader, model, loss_function, nr_features, target):
    valid_total_losses = 0
    model.eval()
    
    with torch.no_grad():
        for batch_number, batch in enumerate(valid_loader, 1):  
            if nr_features == 6:
                X_valid = batch['features'][:,:,:6]
            elif nr_features == 8:
                X_valid = torch.cat((batch['features'][:,:,:6],batch['features'][:,:,8:10]),axis=-1)
            elif nr_features == 9:
                X_valid = torch.cat((batch['features'][:,:,:7],batch['features'][:,:,8:10]),axis=-1)
            else:
                X_valid = torch.cat((batch['features'][:,:,:7],batch['features'][:,:,8:]),axis=-1)
                
            if target == 'gcc':
                y_valid = batch['target'][:,:,0] 
            elif target == 'rcc':
                y_valid = batch['target'][:,:,1] 
            elif target == 'gcc_lowess':
                y_valid = batch['target'][:,:,2]     
            elif target == 'rcc_lowess':
                y_valid = batch['target'][:,:,3]
            elif target == 'gcc_norm':
                y_valid = batch['target'][:,:,4] 
            elif target == 'rcc_norm':
                y_valid = batch['target'][:,:,5] 
            elif target == 'gcc_lowess_norm':
                y_valid = batch['target'][:,:,6]     
            else:
                y_valid = batch['target'][:,:,7]  
            outputs = model(X_valid)
            outputs = outputs[:,365:365+y_valid.shape[-1]]
            loss = loss_function(outputs, y_valid) 
            
            # valid loss
            valid_total_losses += loss.item()
        valid_loss = valid_total_losses / len(valid_loader) 
    return valid_loss


def run_lstm_train(     
        batch_size = 8,
        learning_rate = 0.01,  
        hidden_size = 64, 
        dropout_rate = 0,
        weight_decay = 0,
        patience = 30,
        m = 0,
        pft = 'DB',
        nr_features = 8,
        target = 'gcc_lowess',
        plot_learning_curve = True):
    
    num_layers = 1 
    epochs = 150
    target_size = 1
    input_size = nr_features 
    
    # training the LSTM model and predict target
    train_loss_list = []
    valid_loss_list = []
    training_sets, validation_sets = dl.leave_onesite_out_crossvalidation('../data/datasetm{}/'.format(m), pft, preprocessing = 'normalize', device = device)
    for fold, (train_dataset, valid_dataset) in enumerate(zip(training_sets, validation_sets)):    
        train_loader_shuffle  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_loader_shuffle  = DataLoader(valid_dataset, batch_size = 1, shuffle = True)   
        
        # training the LSTM model
        lstm = LSTM(target_size, input_size, hidden_size, num_layers, dropout_rate)
        lstm = lstm.to(device)
        loss_function = torch.nn.MSELoss()   
        optimizer = torch.optim.AdamW(lstm.parameters(), lr = learning_rate, weight_decay = weight_decay) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9)
        lstm, train_losses, valid_losses, epoch_estop = train_model(train_loader_shuffle, valid_loader_shuffle, lstm, loss_function, optimizer, scheduler, epochs, patience = patience, nr_features = nr_features, target = target)
        train_loss_list.append(train_losses[-1]) 
        valid_loss_list.append(valid_losses[-1])  

        # Save the LSTM model
        torch.save(lstm.state_dict(), "../lstm_models/m{}_{}_{}f_".format(m, pft, nr_features)+ str(fold))   
        
    return 
     

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train an LSTM using shuffled data for one PFT with some of features')
    parser.add_argument('m', help = 'Put the length of blocks to shuffle or "full" to not shuffle')
    parser.add_argument('pft', help = 'The PFT (can be DB, EN or GR)')
    parser.add_argument('nr_features', type = int, help ='6: dynamic variables; 8: dynamic + static climate; 9: dynamic (+ snow) + static climate; 14: dynamic + static climate & soil')
    parser.add_argument('target',  help ='target can be: gcc; rcc; gcc_lowess; rcc_lowess; gcc_norm; rcc_norm; gcc_lowess_norm; rcc_lowess_norm')
    parser.add_argument('batch_size', type = int, help ='DB: batch_size = 8; EN: batch_size = 6; GR: batch_size = 4' )
    args = parser.parse_args()
    
    run_lstm_train(m = args.m, pft = args.pft, nr_features = args.nr_features, target = args.target, batch_size = args.batch_size)
    
    
