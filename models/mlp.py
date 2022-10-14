# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.autograd  import  Function



class MLP(nn.Module):

    def __init__(self, d_input,  num_classes=4):
        super(MLP, self).__init__()
        # parameters
        self.classifier = nn.Sequential(
            nn.Linear(d_input, d_input//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d_input//2, d_input//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d_input//4, num_classes),
        )
        self.avgpool=nn.AdaptiveAvgPool1d(1)

    def forward(self, padded_input):

        out = self.classifier(padded_input)
        out = self.avgpool(out.transpose(-1, -2)).squeeze(dim=-1)

        return out
