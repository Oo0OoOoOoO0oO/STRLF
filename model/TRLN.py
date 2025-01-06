import numpy as np
import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


    
    

class TRLN(nn.Module):
    def Cove2dBlocks(self, dropoutRate, *args, **kwargs):
        conv_block = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1,
                      padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3))
        return conv_block

    def WidelyDepthwiseConv2D(self, wide, dropoutRate,*args,**kwargs):

        block = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, wide), max_norm=1, stride=1, padding=(0, int((wide-1)/2)), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01,
                            affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate)
            )

        return block
        
    def PointConv2D(self,dropoutRate,*args,**kwargs):
        block = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0),
                      groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate),
        )
        return block

    

    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, n_classes),
            nn.Softmax(dim=1))

    def __init__(self, n_classes=4, channels=60, samples=151,
                 dropoutRate=0.5, kernelLength=64, kernelLength2=16, F1=8,
                 D=2, F2=16):
        super(TRLN, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes 
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2 
        self.dropoutRate = dropoutRate
        

        self.convblocks = self.Cove2dBlocks(dropoutRate)
        
        self.widelyDepthwiseConv2D_Block_1 = self.WidelyDepthwiseConv2D(1, dropoutRate)
        
        self.pointConv2DBlock = self.PointConv2D(dropoutRate)
        self.classifierBlock = self.ClassifierBlock(
            144, n_classes)

    def getfeature(self, x):
        x = self.convblocks(x)
        
        x = self.widelyDepthwiseConv2D_Block_1(x)
        x = self.pointConv2DBlock(x)
        x = x.view(x.size()[0], -1)  # Flatten
        
        return x
    
    
    def centorid_loss(self, x, y):
        
        xn = x.data.numpy()
        yn = y.data.numpy()
        x = torch.unsqueeze(x,1)
        
        num_cls = int(np.max(yn))
        
        distance = None
        centorids = []            
        for i in range(num_cls+1):
            x_temp = xn[yn==i]
            centorid = np.mean(x_temp,axis=0)
            centorid = torch.from_numpy(centorid).to(torch.float32)
            centorid = torch.unsqueeze(centorid,0)
            centorids.append(centorid)
        pdist = torch.nn.PairwiseDistance()    
        distance = pdist(x[0],centorids[y[0]])
        
        for i in range(1,x.shape[0]):
            distance = distance + pdist(x[i],centorids[y[i]])
        
        distance = torch.squeeze(distance)
        
        return distance/x.shape[0]
            
    
    def forward(self, x,y):
        x = self.convblocks(x)
        x = self.widelyDepthwiseConv2D_Block_1(x)
        x = self.pointConv2DBlock(x)
        x = x.view(x.size()[0], -1)
        cl = self.centorid_loss(x,y)
        x = self.classifierBlock(x)
        return x, cl
