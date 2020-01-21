import torch
import torch.nn as nn
import torchvision.models as models

# from quantizer.utils import Parser 
from quantizer.quantizer import QuanModel


def hk_fn(module, input, output):
    print("/////////////////////////")
    print(module)

def main():

    model = models.__dict__["inception_v3"](pretrained = True)
    # model = models.__dict__["resnet18"](pretrained = True)
    # input = torch.randn(1, 3, 600,600)
    # trace, out = torch.jit.get_trace_graph(model, args=(input,))
    # torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    # torch_graph = trace.graph()
    # print(out)
    # sys.eixt()
    # for torch_node in torch_graph.nodes():
    #     print(torch_node)
    #     op = torch_node.kind()        
    #     print(op)
        # params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        # print(dir(torch_node))
        # print(torch_node.output())
        # sys.eixt()
        # inputs = [i.unique() for i in torch_node.inputs()]
        # outputs = [o.unique() for o in torch_node.outputs()]
        # print(params)
        # print("Inputs: {}".format(inputs))
        # print("Outputs: {}".format(outputs))

    
    quant_model = QuanModel()
    quant_model.prepare(model)

    # for name, module in model.named_modules():
    #     module.register_forward_hook(hk_fn)
    # x = torch.randn(1, 3,224,224)
    # y = model(x)
    # print(model)



if __name__ == '__main__':
    main()