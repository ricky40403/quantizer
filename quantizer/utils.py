
from collections import OrderedDict
import torch
import torch.nn as nn

import warnings
# import caffe2.python.onnx.frontend.Caffe2Frontend as caffeF

class Graph(object):
    """!
    This is a model graph containing blobs
    """
    def __init__(self):
        self.head = None
        



class Blob(object):
    """!
    This is a blob class, using linked-list type for efficient blob merging.
    """
    def __init__(self, name, layer_type = None, layer = None, params = None):
        self.name = name
        self.type = layer_type
        self.params = params if params else {}
        self.weight = None
        self.bottoms = []
        self.tops =[]
        self.layer = layer
    
    def add_tops(self, top_blob):
        """!
        This function create an interface for tops blobs to add itself to father's tops 
        """
        self.tops.append(top_blob)

    def add_bottoms(self, bottom_blob):
        """!

        """
        self.bottoms.append(bottom_blob)

    def set_bottoms(self, bottoms):
        if not isinstance(bottoms, list):
            warnings.warn("Bottoms should be list, but get {}".format(type(bottoms)))
        self.bottoms = bottoms
    
    def set_tops(self, tops):
        if not isinstance(tops, list):
            warnings.warn("Tops should be list, but get {}".format(type(tops)))

        self.tops = tops
    
    def set_weight(self, weight):
        self.weight = weight

    def get_bottoms_type(self):
        return [bottom.type for bottom in bottoms]

    def get_bottom_byType(self, b_type):
        return [b for b in bottoms if b.type == b_type]
        

class Parser(object):
    """!
    This class parse the pytorch model
    """
    def __init__(self):
        self.mapping_dict = {}
        self.blob_head = Blob("data", "Data", None)
        self.ids = []
        self.tensors = []
        self.param_len = 0
        self.layers = {}

    def check_normal_tensor(self, tensor_id):
        """!
        This function check the tensor belongs to param or normal tensor
        """
        return tensor_id > self.param_len

    def parse(self, model, input):             
        """!
        This function parse the model and build model relation ship
        """

        # reference from torch.onnx's model converting        
        params = list(model.state_dict().values())        
        trace, _ = torch.jit.get_trace_graph(model, args=(input,))
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        torch_graph = trace.graph()
        input_and_param_names = [val.debugName() for val in torch_graph.inputs()] 

        input_len = len(input_and_param_names) - len(params)
        param_names = input_and_param_names[input_len:]
        params_dict = dict(zip(param_names, params))

        # max param id for onnx
        self.param_len = len(param_names)

        # input data id should be 0 ?? (to be checked)
        # creating Graph
        graph= Graph()
        data_blob = Blob("Data")


        # because linked list is hard to directly get the target node
        # using dict to record the node.
        record = {}
        # record = OrderedDict()
        # add input data blobs
        for i in range (input_len):
            data_blob = Blob("Data{}".format(i))
            record[i] = data_blob

        # construct model graph
        for torch_node in torch_graph.nodes():
            print("//////////////  New node ///////////////////////////////////////")
            # remove onnx word
            # print(dir(torch_node))
            op = torch_node.kind().replace("onnx::", "")
            params = {k: torch_node[k] for k in torch_node.attributeNames()} 
            # filter out params' id
            # inputs = [i.unique() for i in torch_node.inputs()]
            inputs = [i.unique() for i in torch_node.inputs() if self.check_normal_tensor(i.unique())]
            # output id  should not contain params
            outputs = [o.unique() for o in torch_node.outputs()]
            # most output should be one, using first id as name
            layer_name = "{}_{}".format(op, outputs[0])
            cur_blob = Blob(layer_name, op, params)


            # create link with father
            for in_ids in inputs:
                
                bottom_blob = record[in_ids]
                cur_blob.add_bottoms(bottom_blob)
                bottom_blob.add_tops(cur_blob)
            
            # record blob
            record[outputs[0]] = cur_blob

            del cur_blob
        

        


# class ConvBn2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, padding_mode = 'zero'):       

#         super(ConvBn2d,self).__init__()



# class ConvBn2dRelu(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, padding_mode = 'zero'): 

#         super(ConvBn2d,self).__init__()


#     def forward(self, x):
#         pass

# class QActivation()

