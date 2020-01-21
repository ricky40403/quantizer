import torch
import torch.nn as nn

from quantizer.utils import Parser


class QuanModel(nn.Module):
    def __init__(self):
        super(QuanModel, self).__init__()

    def prepare(self, model, input = None, quan_type = "int8", per_channel = False):
        """!
        Prepare to quantize the model
        Step1: go through the model and log the layer.
        Step2: find and merge the conv2d, batchnorm2d and relu.
        Step3: quantize tensor after merging.
        """

        model = model.eval()
        if input is None:
            input = torch.randn(1, 3, 600,600)       

        parser = Parser()
        parser.parse(model, input)
            





        pass
        
