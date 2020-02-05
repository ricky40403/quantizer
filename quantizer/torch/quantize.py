import torch
import torch.nn as nn

from quantizer.torch.utils import Parser


def _convert_conv(blob, param, quan_bit = 8, q_per_channel = False):
    """
    This function convert the convolution into pytorch quantized convolution

    @param node: node

    @param param: layer's param

    @quantize: quantize layer or not, default is False
    """
    pass
    

    
def _convert_ConvBNReLU(blob, param, quan_bit = 8, q_per_channel = False):
    pass



QUAN_DICT = {
    "Conv": _convert_conv,
    # "BatchNormalization": _convert_BatchNorm,
    # "Relu": _convert_relu,
    # "ConvTranspose": _convert_conv_transpose,
    # "Add": _convert_Add,
    # "Mul": _convert_Mul,
    # "Reshape": _convert_Reshape,
    # "MaxPool": _convert_pool,
    # "AveragePool": _convert_pool,
    # "Dropout": _convert_dropout,
    # "Gemm": _convert_gemm,
    # "Upsample": _convert_upsample,
    # "Concat": _convert_concat,    
    # "Sigmoid": _convert_sigmoid,
    # "Flatten": _convert_Flatten,
    # "ConvBN": _convert_ConvBn,
    "ConvBnReLU": _convert_ConvBNReLU,
}
