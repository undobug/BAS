# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


from .mlp import MLP
from .models_transformer_baseline import SER_search

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'search':
        size = 1024
        layers = 4
        model =  SER_search(key_size=size, query_size=size, value_size=size, num_hiddens=size,
                norm_shape=[config.DATA.WAV_LENGTH, size], ffn_num_input=size, ffn_num_hiddens=int(size * 0.5), num_heads=16,
                num_layers=layers,
                dropout=0.5,num_classes=config.MODEL.NUM_CLASSES)

    elif model_type == 'mlp':
        model = MLP(d_input=1024, num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
