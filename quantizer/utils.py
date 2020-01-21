
import torch
import torch.nn as nn


class Blobs(object):
    """!
    This is a blob class
    """
    def __init__(self, name, layer_type, params = None, bottoms = [], tops = []):
        self.name = name
        self.type = layer_type
        self.params = params if params else {}
        self.weight = None
        self.bottoms = bottoms
        self.tops = tops
    
    def set_bottoms(self, bottoms):
        self.bottoms = bottoms
    
    def set_tops(self, tops):
        self.tops = tops
    
    def set_weight(self, weight):
        self.weight = weight


class Parser(object):
    """!
    This class parse the pytorch model
    """
    def __init__(self):
        self.mapping_dict = {}
        self.blob_head = Blobs("data", "Data", None)
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
        param_names = input_and_param_names[len(input_and_param_names) - len(params):]
        params_dict = dict(zip(param_names, params))

        # max param id for onnx
        self.param_len = len(param_names)
        
        # construct model graph
        for torch_node in torch_graph.nodes():
            # remove onnx word
            op = torch_node.kind().replace("onnx::", "")
            params = {k: torch_node[k] for k in torch_node.attributeNames()} 
            # filter out params' id
            inputs = [i.unique() for i in torch_node.inputs() if self.check_normal_tensor(i.unique())]
            # output id  should not contain params
            outputs = [o.unique() for o in torch_node.outputs()]
            
            
            blob = Blobs("QQ", op, params)
            
            print(op)
            print(inputs)
            print(outputs)
            



        

        


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

