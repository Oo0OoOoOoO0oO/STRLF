import torch
import torch.nn as nn
import torch.nn.functional as F

class FcNet(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden, n_output):
        super(FcNet, self).__init__()    
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output)  
        
        self.elu = nn.ReLU()
 
    def forward(self, x):  
        x = self.elu(self.hidden1(x))    
        x = self.predict(x)       
        return x

