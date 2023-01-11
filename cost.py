"""
Requirements:
    pip install ptflops
"""
import torch
from ptflops import get_model_complexity_info
from models.model.LMNet import LMNet


def input_constructor(input_shape):
    inputs = {'L': torch.ones(input_shape), 'R': torch.ones(input_shape)}
    return inputs


with torch.cuda.device(0):
    model = LMNet(192)
    macs, params = get_model_complexity_info(model, (1, 3, 256, 512), as_strings=True, print_per_layer_stat=True,
                                             verbose=False, input_constructor=input_constructor)
    print('{:<30}  {:<8}'.format('Number of operations: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
