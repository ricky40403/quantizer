import torch
import torch.nn as nn
import torchvision.models as models


def main():

    model = models.__dict__["resnet18"](pretrained = True)
    print(model)



if __name__ == '__main__':
    main()