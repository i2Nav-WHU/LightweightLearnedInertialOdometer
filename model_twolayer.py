from math import exp
import os
import sys

sys.path.append(os.path.dirname(__file__))

import einops
import torch
from torch import nn

from einops.layers.torch import *
from torch.nn.modules.dropout import Dropout

from model_MLP import *

class ResMLPExtractor(nn.Module):
    def __init__(self, patch_num=10,
                 patch_len=10,
                 input_channel=6,
                 mlp_in_dim=15,
                 expansion=4,
                 active_func=nn.GELU(),
                 layer_num=4,
                 dropout=0.5):
        '''
        INPUT: [Batch_num, input_dim(6 for 6-axis IMU), measurement_length]
        OUTPUT: [Batch_num, patch_num, feature_length]
        '''
        super(ResMLPExtractor, self).__init__()

        def wrapper(i, fn): return PreAffinePostLayerScale(mlp_in_dim, i, fn)

        self.net = nn.Sequential(
            ##### Feature Convert
            Rearrange('b c (l w) -> b l (w c)', w=patch_len),
            nn.Linear(int(patch_len * input_channel), mlp_in_dim),
            #### ResMLP Module
            *[
                nn.Sequential(
                    wrapper(i, nn.Conv1d(patch_num, patch_num, 1, bias=False)),
                    wrapper(i, nn.Sequential(
                        nn.Linear(mlp_in_dim, mlp_in_dim *
                                  expansion, bias=False),
                        active_func,
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(mlp_in_dim * expansion,
                                  mlp_in_dim, bias=False)
                    ))
                ) for i in range(layer_num)
            ],
            Affine(mlp_in_dim)
        )

    def forward(self, x):
        return self.net(x)


class ResLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(x) + x




class SimpleMLPReg(nn.Module):
    def __init__(self, patch_num=10, feature_len=10, out_dim=3, layer_num=4, active_fun=nn.GELU(), dropout=0.5):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(SimpleMLPReg, self).__init__()
        self.input_len = int(patch_num * feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Rearrange('b l f -> b (l f)'),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
            # nn.Linear(self.input_len, out_dim * 2)
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


class PoolingMLPReg(nn.Module):
    def __init__(self, patch_num=10,
                 feature_len=10,
                 out_dim=3,
                 layer_num=4,
                 active_fun=nn.GELU(),
                 dropout=0.5,
                 pooling_type='mean'):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(PoolingMLPReg, self).__init__()
        self.input_len = int(feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Reduce('b l f-> b f', pooling_type),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


'''
# Extractor exmaples:
                "extractor": {
                    "name": "ResMLP",
                    "layer_num": 4,
                    "expansion": 4,
                },
# Reg examples:
                "reg": {
                    "name": "MLP",
                    "layer_num": 3,
                }
                "reg":{
                    "name":"EAMLP",
                    "layer_num":3,
                    "inner_dim":64
                }
                
'''


class TwoLayerModel(nn.Module):
    def __init__(self, model_para=None):
        super(TwoLayerModel, self).__init__()

        if model_para is None:
            model_para = {
                "input_len": 100,
                "input_channel": 6,
                "patch_len": 10,
                "feature_dim": 20,
                "out_dim": 3,
                "active_func": "GELU",
                "extractor": {
                    "name": "ResMLP",
                    "layer_num": 4,
                    "expansion": 4,
                },
                "reg": {
                    "name": "MLP",
                    "layer_num": 3,
                }

            }

        self.active_function = None
        if model_para["active_func"] == "GELU":
            self.active_function = nn.GELU()
        elif model_para["active_func"] == "ReLU":
            self.active_function = nn.ReLU()
        else:
            self.active_function = nn.GELU()

        patch_num = int(model_para["input_len"] / model_para["patch_len"])
        if model_para["extractor"]["name"] == "ResMLP":
            self.extractor = ResMLPExtractor(patch_num=patch_num, patch_len=model_para["patch_len"],
                                             input_channel=model_para["input_channel"],
                                             mlp_in_dim=model_para["feature_dim"],
                                             expansion=model_para["extractor"]["expansion"],
                                             active_func=self.active_function,
                                             layer_num=model_para["extractor"]["layer_num"],
                                             dropout=model_para["extractor"]["dropout"]
                                             )

        else:
            print("unknown extractor name:", model_para["extractor"]["name"])

        if model_para["reg"]["name"] == "MLP":
            self.reg = SimpleMLPReg(patch_num=patch_num, feature_len=model_para["feature_dim"],
                                    layer_num=model_para["reg"]["layer_num"], active_fun=self.active_function)
        elif model_para['reg']['name'] == "MaxMLP":
            self.reg = PoolingMLPReg(patch_num=patch_num,

                                     out_dim=model_para['out_dim'],
                                     feature_len=model_para['feature_dim'],
                                     layer_num=model_para['reg']['layer_num'],
                                     active_fun=self.active_function,
                                     pooling_type='max')
        elif model_para['reg']['name'] == 'MeanMLP':
            self.reg = PoolingMLPReg(patch_num=patch_num, feature_len=model_para['feature_dim'],
                                     out_dim=model_para['out_dim'],
                                     layer_num=model_para['reg']['layer_num'], active_fun=self.active_function,
                                     pooling_type='mean')
        else:
            print("unknown reg name: ", model_para['reg']['name'])

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature = self.extractor(x)
        out1, out2 = self.reg(feature)
        return out1, out2


if __name__ == '__main__':
    # Test Transformer

    # *|CURSOR_MARCADOR|*
    #
    #
    #
    model_para = {
        "input_len": 100,
        "input_channel": 6,
        "patch_len": 25,
        "feature_dim": 512,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": { # include: Feature Convert & ResMLP Module in the paper Fig. 3.
            "name": "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.2,
        },
        "reg": { # Regression in the paper Fig.3
            "name": "MeanMLP",
            # "name": "MaxMLP",
            "layer_num": 3,
        }
    }

    net = TwoLayerModel(model_para) # initialize the model
    x = torch.rand([512, 6, 100]) # batch_size, input_channel, input_len
    '''
    For example: 
    1 seconds of imu output at 100Hz, we got [6 x 100] matrix.
    '''

    y, y_cov = net(x) # output: [batch_size, 3], [batch_size, 3]
    print('x:', x.shape, 'y', y.shape, 'y_cov', y_cov.shape)
