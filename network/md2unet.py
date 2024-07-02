import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class _MDBLayer(nn.Module):
    def __init__(self,
                 input_c: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 stride=1,
                 scale=0.05,
                 memory_efficient: bool = False):
        super(_MDBLayer, self).__init__()
        inter_c = input_c //3
        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.LeakyReLU(inplace=True))
        self.add_module("branch0",nn.Sequential(
                                                      BasicConv(input_c, inter_c, kernel_size=1, stride=stride),
                                                      BasicConv(inter_c, growth_rate, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        ))
        self.add_module("branch1",nn.Sequential(
                                                      BasicConv(input_c, inter_c, kernel_size=1, stride=1),
                                                      BasicConv(inter_c, growth_rate, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
                                                      BasicConv(growth_rate, growth_rate, kernel_size=3, stride=1, padding=2, dilation=2, relu=False)
        ))
        self.add_module("branch2", nn.Sequential(
                                                      BasicConv(input_c, inter_c, kernel_size=1, stride=1),
                                                      BasicConv(inter_c, growth_rate, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
                                                      BasicConv(growth_rate, growth_rate, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        ))
        self.add_module("ConvLinear",BasicConv(3*growth_rate, growth_rate, kernel_size=1, stride=1, relu=False))
        self.add_module("shortcut",BasicConv(input_c, growth_rate, kernel_size=1, stride=stride, relu=False))
        self.add_module("relu2",nn.LeakyReLU(inplace=False))

        self.scale = scale
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concat_features = torch.cat(inputs, 1)
        x0 = self.branch0(self.relu1(self.norm1(concat_features)))
        x1 = self.branch1(self.relu1(self.norm1(concat_features)))
        x2 = self.branch2(self.relu1(self.norm1(concat_features)))
        # print("concat size:",concat_features.size())
        # print("x0 size:", x0.size())
        # print("x1 size:", x1.size())
        # print("x2 size:", x2.size())

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(concat_features)
        new_features = out + short*self.scale
        new_features = self.relu2(new_features)

        return new_features

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint(self, inputs: List[Tensor]) -> Tensor:
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            new_features = self.call_checkpoint(prev_features)
        else:
            new_features = self.bn_function(prev_features)
        if self.drop_rate > 0:

            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _MDBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_MDBlock, self).__init__()
        for i in range(num_layers):
            layer = _MDBLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("MDBLayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.LeakyReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c,
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class MD2UNet(nn.Module):
    """
    MD2Unet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each mdblayer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """

    def __init__(self,
                 growth_rate: int = 48,
                 block_config: Tuple[int, int, int, int] = (4, 8, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0.25,
                 num_classes: int = 3,
                 memory_efficient: bool = False):
        super(MD2UNet, self).__init__()

        # first conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.LeakyReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # each MDBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _MDBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("MDBlock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.Leakyrelu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def MD2UNet161(**kwargs: Any) -> MD2UNet:

    return MD2UNet(growth_rate=48,
                    block_config=(4, 8, 24, 16),
                    num_init_features=96,
                    **kwargs)

def load_state_dict(model: nn.Module, weights_path: str) -> None:
    # '.'s are no longer allowed in module names, but previous _MDBLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*mdblayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(weights_path)

    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)
    print("successfully load pretrain-weights.")
