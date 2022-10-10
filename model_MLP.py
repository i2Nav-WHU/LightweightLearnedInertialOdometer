from numpy.core.fromnumeric import transpose
from numpy.lib.arraypad import pad
from numpy.lib.arraysetops import isin
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d

import torch


from einops.layers.torch import *


class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super(PreAffinePostLayerScale, self).__init__()
        init_eps = 0.1
        if depth <= 18:
            init_eps = 0.1
        else:
            init_eps = 1e-5

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.affine_out = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.affine_out(self.fn(self.affine(x)) * self.scale + x)


class ResMLP(nn.Module):
    def __init__(self, dim, expansion_factor=4, depth=1, active_function=None):
        super(ResMLP, self).__init__()

        self.res_mlp = PreAffinePostLayerScale(
            dim, depth,
            nn.Sequential(
                nn.Linear(dim, int(dim * expansion_factor)),
                active_function,
                nn.Linear(int(dim * expansion_factor), dim)
            )
        )

    def forward(self, x):
        return self.res_mlp(x)


class MLPExtractor(nn.Module):
    def __init__(self, in_channel_len=20,
                 in_channel_dim=6,
                 out_channel=20,
                 res_layer_num=1,
                 inner_dim=None,
                 active_func=None):
        super(MLPExtractor, self).__init__()

        if inner_dim is None:
            inner_dim = [60, 30]

        self.in_channel = in_channel_dim * in_channel_len
        self.out_channel = out_channel

        self.resmlp_list = list()
        for i in range(res_layer_num):
            self.resmlp_list.append(ResMLP(self.in_channel, 4, i, active_func))

        self.resmlp_list = nn.ModuleList(self.resmlp_list)

        self.mlp_list = list()
        self.mlp_list.append(nn.Linear(self.in_channel, inner_dim[0]))
        for i in range(1, len(inner_dim)):
            self.mlp_list.append(nn.Linear(inner_dim[i - 1], inner_dim[i]))

        self.mlp_list = nn.ModuleList(self.mlp_list)

        self.output_layer = nn.Linear(inner_dim[-1], out_channel)

        self.active_function = active_func
        self.dropout = nn.Dropout(0.5)

        self.__initialization()

    def forward(self, x):  # torch.Tensor):
        out = x
        for layer in self.resmlp_list:
            out = layer(out)
        for layer in self.mlp_list:
            out = layer(out)
            out = self.active_function(out)
            out = self.dropout(out)
        out = self.output_layer(out)
        return out

    def __initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.xavier_normal(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MLPReg(nn.Module):
    def __init__(self, in_size,
                 out_dim=3,
                 inner_dims=[80, 50, 20],
                 res_net_layer=4,
                 active_fun=None,
                 batch_norm=None,
                 dropout=0.5):
        super(MLPReg, self).__init__()

        # self.in_dim = in_dim
        # self.in_num = in_num
        self.in_size = in_size  # self.in_dim * self.in_num
        self.out_size = out_dim * 2

        self.resmlp_list = list()
        for i in range(res_net_layer):
            self.resmlp_list.append(ResMLP(self.in_size, 4, i, active_fun))
        self.resmlp_list = nn.ModuleList(self.resmlp_list)

        self.mlp_list = list()
        self.bn_list = list()
        self.mlp_list.append(nn.Linear(int(self.in_size), int(inner_dims[0])))
        self.bn_list.append(nn.BatchNorm1d(int(inner_dims[0])))

        for i in range(1, len(inner_dims)):
            self.mlp_list.append(nn.Linear(int(inner_dims[i - 1]), int(inner_dims[i])))
            self.bn_list.append(nn.BatchNorm1d(int(inner_dims[i])))

        self.mlp_list = nn.ModuleList(self.mlp_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.output_layer = nn.Linear(int(inner_dims[-1]), int(self.out_size))

        self.active_func = active_fun
        # self.bn = nn.BatchNorm1d()
        self.dropout = nn.Dropout(dropout)

        self.__initialization()

    def forward(self, x):
        out = x.reshape([x.size(0), 1, x.size(1)])

        # processing
        for layer in self.resmlp_list:
            out = layer(out)

        out = out.reshape([x.size(0), x.size(1)])

        for i in range(len(self.mlp_list)):
            out = self.mlp_list[i](out)

            out = self.active_func(out)
            out = self.bn_list[i](out)
            out = self.dropout(out)

        out = self.output_layer(out)
        return out[:, 0:3], out[:, 3:6]

    def __initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MLPCombineNet(nn.Module):
    def __init__(self, para=None):
        super(MLPCombineNet, self).__init__()
        if para is None:
            para = {
                "input_len": 100,
                "in_channel_len": 20,
                "in_channel_dim": 6,
                "out_channel": 20,
                "res_layer_num": 4,
                "reg_res_layer_num": 4,
                "inner_dim": [60, 30],
                "active_function": "ReLU",  # "GELU",
                "reg_inner_dims": [80, 50, 20],
                "batch_norm": None,
                "dropout": 0.5,
                "out_dim": 3
            }

        self.in_channel_len = para["in_channel_len"]
        self.in_channel_dim = para["in_channel_dim"]
        self.active_function = None
        self.active_func_name = para["active_function"]
        if self.active_func_name == "GELU":
            self.active_function = nn.GELU()
        elif self.active_func_name == "ReLU":
            self.active_function = nn.ReLU(inplace=True)
        elif self.active_func_name == "PReLU":
            self.active_function = nn.PReLU()
        else:
            print('active function name unknown[{0}]'.format(self.active_func_name))
        self.extractor = MLPExtractor(in_channel_len=para["in_channel_len"],
                                      in_channel_dim=para["in_channel_dim"],
                                      out_channel=para["out_channel"],
                                      res_layer_num=para["res_layer_num"],
                                      inner_dim=para["inner_dim"],
                                      active_func=self.active_function)

        self.reg_input_size = int(para["out_channel"] * para["input_len"] / para["in_channel_len"])
        print('reg input size:', self.reg_input_size)
        self.reg = MLPReg(
            self.reg_input_size,
            active_fun=self.active_function,
            res_net_layer=para["reg_res_layer_num"],
            batch_norm=para["batch_norm"],
            dropout=para["dropout"]
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        out = x.reshape([x.size(0), int(x.size(1) / self.in_channel_len), -1, self.in_channel_dim])
        out = torch.flatten(out, 2)
        out = self.extractor(out)
        out = out.reshape([x.size(0), -1])
        out, out_cov = self.reg(out)
        return out, out_cov
