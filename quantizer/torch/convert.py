import torch
from quantizer.torch.utils import Parser
import quantizer.torch.quantize as quan


class Converter(object):
    """!
    This class convert the layer and generate the quantization model
    """

    def __init__(self, quan_bit = 8):
        self.quan_bit = quan_bit
        self.graph = None

    def prepare(self, model, input = None, quan_type = "int8", per_channel = False):
        """!
        Prepare to quantize the model
        Step1: go through the model and log the layer.
        Step2: find and merge the conv2d, batchnorm2d and relu.
        Step3: quantize tensor after merging.
        """

        # Step1: go through the model and log the layer.
        model = model.eval()
        if input is None:
            input = torch.randn(1, 3, 600,600)
            
        parser = Parser()
        graph = parser.parse(model, input)
        graph.merge(["Conv", "BatchNormalization"], "ConvBn")
        graph.merge(["ConvBn", "Relu"], "ConvBnReLU")

        self.graph = graph

        for ids in graph.nodes.keys():
            print("=========================================================")
            blob = graph.get_node(ids)
            print("Blob ids: {}".format(ids))
            print("Blob: {}".format(blob))
            print("Blob name: {}".format(blob.name))
            print("Blob type : {}".format(blob.type))
            print("Blob param : {}".format(blob.params))
            print("Blob bottoms: {}".format([b.name for b in blob.bottoms]))
            print("Blob tops: {}".format([b.name for b in blob.tops]))


    def convert(self, quan_bit = 8, quan_per_channel = False):
        """!
        This function create quantized layer from graph
        return quantization model
        """

        if self.graph is None:
            raise ValueError("You need to prepare a model using prepare(model)")

        
        print(" ==> Quantize bit: {} bit".format(quan_bit))
        print(" ==> Quantize per channel: {}.".format(quan_per_channel))


        
        for ids in self.graph.nodes.keys():
            blob = self.graph.get_node(ids)
            if blob.type is None:
                continue
            print(blob.type)
            print(quan.QUAN_DICT[blob.type])
            exit()
            tmp_module = quan.QUAN_DICT[blob.type](blob)





            
        


            
        








