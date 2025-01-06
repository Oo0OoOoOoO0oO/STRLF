import numpy as np
import torch
from torch import nn
from BiMap import BiMap
from LogEig import LogEigFunction
from ReEig import ReEigFunction
from utils import getCov
import pyriemann.utils.mean as Rmeans


class SRLN(nn.Module):


    def __init__(self):
        super(SRLN, self).__init__()
        self.bimap1 = BiMap(9, 9)
        self.bimap2 = BiMap(9, 9)
        self.bimap3 = BiMap(9, 9)
        self.linear = nn.Linear(9, 2)

    def forward(self, x, y):
        x = self.bimap1(x)
        x = ReEigFunction.apply(x, 1e-4)
        x = self.bimap2(x)
        x = ReEigFunction.apply(x, 1e-4)
        x = self.bimap3(x)
        
        cl = self.compute_centerLoss(x,y)
        
        x = LogEigFunction.apply(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x,cl
    
        
    
    def compute_centerLoss(self, x, y):
        xn = x.data.numpy()
        yn = y.data.numpy()
        
        num_cls = int(np.max(yn))
        
        distance = None            
        for i in range(num_cls+1):
            x_temp = xn[yn==i]
            try:
                centroid = Rmeans.mean_covariance(x_temp,metric='riemann')
            except Exception as ex:
                print("error: %s"%ex)
                continue
            centroid = torch.from_numpy(centroid).to(torch.float32)
            
            x_temp = x[y==i]
            if distance == None:
                distance = jeffrey_divergence(x_temp[0],centroid)
            else:
                distance += jeffrey_divergence(x_temp[0],centroid)
            
            for sample in x_temp[1:]:
                
                # compute the distance between a sample and the class centroid
                jd = jeffrey_divergence(sample,centroid)
                distance += jd
            
        return distance/x.shape[0]
            
        