
import torch
import torch.nn as nn


class Blobs(object):
    """!
    This is a blob class
    """
    def __init__(self, name, bottoms = [], tops = []):
        self.name = name
        self.bottoms = bottoms
        self.tops = tops
    
    def set_bottoms(self, bottoms):
        self.bottoms = bottoms
    
    def set_tops(self, tops):
        self.tops = tops


class Parser(object):
    """!
    This class parse the pytorch model
    """
    def __init__(self):
        pass

    def parse(self, model):
        
        tmp_tensor = torch.FloatTensor([3, 224, 224])

        

    





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

