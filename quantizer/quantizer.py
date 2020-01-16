import torch
import torch.nn as nn


class QuanModel(nn.Moodule):
    def __init__(self):

        super(self, QuanModel).__init__()
        


    def prepare(self, model, quan_type = "int8", per_channel = False):
        """!
        Prepare to quantize the model
        Step1: go through the model and log the layer.
        Step2: find and merge the conv2d, batchnorm2d and relu.
        Step3: quantize tensor after merging.
        """



        pass
        
