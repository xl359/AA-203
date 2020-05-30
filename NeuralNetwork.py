#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:11:52 2020

@author: xinweiliu
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

#this class is necessary for getting elements in an object

class TrickyFunctionDataset(Dataset):
    def __init__(self, num_samples=2000):
        self.n = num_samples
        self.generate_data()
        
        
    def generate_data(self):
        # sample x
        self.X = np.zeros(self.n)
        self.Y = np.zeros(self.n)
        for i in range(self.n):
            x = np.random.rand()*10-5
            y = 2./(1 + np.exp(-x**2)) + 0.25*np.cos(x) + 0.1*np.random.randn()
            
            self.X[i] = x
            self.Y[i] = y
        
    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        return(self.X[idx:idx+1], self.Y[idx:idx+1])
        
function_dataset = TrickyFunctionDataset()
function_dataloader = DataLoader(function_dataset, batch_size=20, shuffle=True, num_workers=4)



class FeedforwardNetwork(nn.Module):
    def __init__(self, hidden_dim1=32, hidden_dim2=32):
        super().__init__()
        #set up first layer
        self.fc1 = nn.Linear(1,hidden_dim1)
        self.activation = nn.ReLU()
        #set up second layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        #set uo third layer
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
    def forward(self,x):
        #apply layer and activation
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        return self.fc3(h2)



def mse_loss(y,ypred):
    return torch.mean((y-ypred)**2)

#load neural network

model = FeedforwardNetwork().double()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
for epoch in range(num_epochs):
    for i, (x,y) in enumerate(function_dataloader):   
     #pass model through neural network
        ypred = model(x)
        loss = mse_loss(y,ypred)
        
        #need to zero the gradient
        optimizer.zero_grad()
        loss.backward()
        #gradient descent
        optimizer.step()
        
    print('Epoch {}; Loss = {}'.format(epoch, loss.item()))
    
    
x_data = [] 
y_data = [] 

for i in range(function_dataset.n):
    
    #here function_dataset 
   # is an object. To access the element, need to use getitem
    #to get an element in the object, define a function that will help to get elements
    x = function_dataset.__getitem__(i)[0].item()
    y = function_dataset.__getitem__(i)[1].item()
    x_data.append(x)
    y_data.append(y)

plt.scatter(x_data,y_data,alpha=0.15)

x_viz = np.linspace(-5,5,300)
#passing through model will yeild y, this value is passed through optimizer
y_viz = [model(torch.tensor([x])).item() for x in x_viz]
plt.plot(x_viz,y_viz, color='r')
plt.show()





