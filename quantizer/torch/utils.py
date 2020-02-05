import warnings
from collections import OrderedDict
import torch
import torch.nn as nn


class Graph(object):
    """!
    This is a model graph containing blobs
    """
    def __init__(self):
        self.input_ids = []
        self.nodes = OrderedDict()
    
    def set_input_node(self, id, node):
        self.input_ids.append(id)
        self.add_node(id, node)
    
    def add_node(self, id, node):
        self.nodes[id] = node

    def get_node(self, id):
        return self.nodes[id]

    def get_all_type(self):
        return list(set([self.nodes[k].type for k in self.nodes]))

    def merge(self, source_list, target, verbose = False):
        """!
        This function merge the layers in source_list to target layer
        """
        # to do (type handling)
        # input must be list
        if not isinstance(source_list, list):
            warnings.warn("Warning, source must be list, but get {}".format(type(source_list)))
            source_list = [source_list]
        
        seened_id = []
        # print("/////////////////////////////")
        verbose_str = ""
        node_to_remove = []
        for key in self.nodes.keys():

            if key in seened_id:
                continue

            cur_blob = head_blob  = tail_blob = self.nodes[key]

            # print("Cur Key : {}".format(head_blob.name))
            # print("Cur Blob: {}".format(head_blob))

            # current only support merge layer that has only one output
            if len(cur_blob.tops) > 1:
                seened_id.append(key)
                continue

            tmp_nodesList = []
            tmp_paramList = []
            tmp_weightList = []
            for s in source_list:
                # current only support merge layer that has only one output
                # if len(cur_blob.tops) > 1:
                #     break

                tail_blob = cur_blob

                if cur_blob.type == s:
                    # print("Found mateched : {}".format(cur_blob.name))
                    tmp_nodesList.append(cur_blob)
                    tmp_paramList.extend(cur_blob.params)
                    tmp_weightList.extend(cur_blob.weight)
                    cur_blob = cur_blob.tops[0]
                
                

            # all match
            if len(tmp_nodesList) == len(source_list):
                
                new_blob = Blob(key, "{}{}".format(target, key), target, tmp_paramList)
                new_blob.set_weight(tmp_weightList)
                new_blob.set_bottoms(head_blob.bottoms)
                # replace the tops relation of the head blob's bottoms
                for bottom in head_blob.bottoms:
                    # print(bottom)
                    top_list = [new_blob if b == head_blob else b for b in bottom.tops ]
                    bottom.set_tops(top_list)
                # replace the botooms relation of the tail blob's top
                # cur_blob is tail's top now, so use cur_blob
                for top in tail_blob.tops:
                    bottom_list = [new_blob if b == tail_blob else b for b in top.bottoms]
                    top.set_bottoms(bottom_list)

                # top should be one, set cur_blob
                new_blob.set_tops([cur_blob])
                self.nodes[key] = new_blob
                
                for tmp_node in tmp_nodesList[1:]:                    
                    node_to_remove.append(tmp_node)
                    seened_id.append(tmp_node.key)
                    # del tmp_node


        # remove nodes
        for node in node_to_remove:
            del self.nodes[node.key]
                
        # for ids in self.nodes.keys():
        #     blob = self.get_node(ids)
        #     print("Blob ids: {}".format(ids))
        #     print("Blob: {}".format(blob))
        #     print("Blob name: {}".format(blob.name))
        #     print("Blob bottoms: {}".format(blob.bottoms))
        #     print("Blob tops: {}".format(blob.tops))

        
        

            
        


        



class Blob(object):
    """!
    This is a blob class, using linked-list type for efficient blob merging.
    """
    def __init__(self, key, name, layer_type = None, layer = None, params = None):
        self.key = key
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

        if not isinstance(weight, list):
            warnings.warn("weight should be list, but get {}".format(type(weight)))
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
        self.param_names = []

    def check_normal_tensor(self, tensor_id):
        """!
        This function check the tensor belongs to param or normal tensor
        """
        return tensor_id not in self.param_names

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
        self.param_names = [int(i) for i in input_and_param_names[input_len:]] 
        params_dict = dict(zip(self.param_names, params))

        # input data id should be 0 ?? (to be checked)
        # creating Graph
        graph= Graph()
        # data_blob = Blob("Data")

        # because linked list is hard to directly get the target node        
        for i in range (input_len):
            # data_key_name = input_and_param_names[i]
            data_blob = Blob(i, "Data{}".format(i))
            # record[i] = data_blob
            graph.set_input_node(i, data_blob)

        
        # construct model graph
        for torch_node in torch_graph.nodes():
            # print("//////////////  New node ///////////////////////////////////////")            
            # remove onnx word            
            op = torch_node.kind().replace("onnx::", "")
            params = {k: torch_node[k] for k in torch_node.attributeNames()} 
            # filter out params' id
            inputs = [i.unique() for i in torch_node.inputs() if self.check_normal_tensor(i.unique())]            
            params = [i.unique() for i in torch_node.inputs() if not self.check_normal_tensor(i.unique())]
            # output id  should not contain params
            outputs = [o.unique() for o in torch_node.outputs()]
            # use first id as key
            key = outputs[0]
            layer_name = "{}_{}".format(op, key)
            cur_blob = Blob(key, layer_name, op, params)
            
            weights = []
            for w in params:
                weights.append(params_dict[w])
            
            cur_blob.set_weight(weights)

            # print(op)
            # print(params)
            # print(inputs)
            # print(outputs)



            # create link with father
            for in_ids in inputs:                
                bottom_blob = graph.get_node(in_ids)
                # add relationship
                cur_blob.add_bottoms(bottom_blob)
                bottom_blob.add_tops(cur_blob)            
            
            graph.add_node(outputs[0], cur_blob)

            del cur_blob

        # for ids in graph.nodes.keys():
        #     blob = graph.get_node(ids)
        #     print("Blob ids: {}".format(ids))
        #     print("Blob: {}".format(blob))
        #     print("Blob name: {}".format(blob.name))
        #     print("Blob bottoms: {}".format(blob.bottoms))
        #     print("Blob tops: {}".format(blob.tops))

        return graph
        
        
        


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

