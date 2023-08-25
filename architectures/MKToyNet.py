import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class _MKToyNet(nn.Module):
    def __init__(self, input_size=32, num_init_features=64,  num_classes=10):

        super(_MKToyNet, self).__init__()
        self.l1 = nn.Linear(input_size*input_size, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,10)
        self.do = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()



    def forward(self, x):
        b = x.size(0)     
        x = x.view(b,-1)
        
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2+h1)
        logits = self.l3(do)
        return logits
    

def MKToyNet(**kwargs):
    # Mohammad Khaloooei(MK) 's Toy Network (This template is used for all architectures)
    # default values are for cifar10 #num_init_features=24, input_size=32, 
    model = _MKToyNet(**kwargs)
    return model