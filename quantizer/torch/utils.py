
import torch
import torch.nn as nn



class ConvBn2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode = 'zero'):       

        super(ConvBn2d,self).__init__()



class ConvBn2dRelu(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode = 'zero'): 

        super(ConvBn2d,self).__init__()


    def forward(self, x):
        pass

class QActivation()

