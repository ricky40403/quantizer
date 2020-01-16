import torch
import torch.nn as nn


class QuanModel(nn.Moodule):
    def __init__(self):        
        super(self, QuanModel).__init__()
        


    def prepare(self, model, quan_type = "int8", per_channel = False):
        pass
        
