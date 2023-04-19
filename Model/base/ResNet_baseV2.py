from Model.base.ResNet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn
from typing import List
from Model.base.layers import (
    InvertedBlock,
    OpSequential,
    ResidualBlock,
    SeInvertedBlock,
)
from Model.base.layers import (
    ConvLayer,
    ConvLayer_padding,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
    SeInvertedBlock,
    PoolLayer,
    BasicBlock_base,
    BasicBlock_baseV2,
)



class ResNet_base(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, stage_width_list=[64, 128, 256, 512], num_channels=3):
        super(ResNet_base, self).__init__()
        input_block = OpSequential(
            [
                ConvLayer_padding(num_channels, stage_width_list[0], kernel_size=7, stride=2, padding=3, use_bias=False),
                PoolLayer(kernel_size=3, stride=2, padding=1)
            ]
        )
        block1 = self._make_layer(ResBlock, layer_list[0], stage_width_list[0], stage_width_list[0])
        block2 = self._make_layer(ResBlock, layer_list[1], stage_width_list[0], stage_width_list[1], stride=2)
        block3 = self._make_layer(ResBlock, layer_list[2], stage_width_list[1], stage_width_list[2], stride=2)
        block4 = self._make_layer(ResBlock, layer_list[3], stage_width_list[2], stage_width_list[3], stride=2)

        FChead = OpSequential(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                LinearLayer(
                    in_features=stage_width_list[3],
                    out_features=num_classes
                )
            ]
        )
        self.backbone = nn.ModuleDict(
            {
                "input_block": input_block,
                "block": nn.ModuleList([block1, block2, block3, block4]),
            }
        )
        self.head = FChead

    def forward(self, x):
        x = self.backbone['input_block'](x)
        #print(x.shape)
        for i, block in enumerate(self.backbone['block']):
            x = block(x)
            #print(x.shape)
        x = self.head(x)
        #print(x.shape)
        return x

    def _make_layer(self, ResBlock, blocks, in_channels, out_channel, stride=1):
        layers = []
        layers.append(ResBlock(conv1 = ConvLayer_padding(in_channels, out_channel, kernel_size=3, padding=1, stride=stride, use_bias=False),
                               conv2 = ConvLayer_padding(out_channel, out_channel, kernel_size=3, padding=1, stride=1, use_bias=False, act_func=None),
                               downsample = ConvLayer_padding(in_channels, out_channel, kernel_size=1, padding=0, stride=stride, use_bias=False, act_func=None)))
        self.in_channels = out_channel * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(
                conv1=ConvLayer_padding(self.in_channels, out_channel, kernel_size=3, padding=1, stride=1,
                                        use_bias=False),
                conv2=ConvLayer_padding(out_channel, out_channel, kernel_size=3, padding=1, stride=1,
                                        use_bias=False, act_func=None),
                downsample=None))

        return OpSequential(layers)


def count_grad_parameters(model):
    """
    Counts the number of parameters that have a non-zero gradient value.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: The number of parameters with non-zero gradients.
    """
    device = next(model.parameters()).device
    num_grad_params = torch.zeros(1, device=device)
    for param in model.parameters():
        if param.grad is not None:  # Check if the gradient is not None.
            num_grad_params += torch.count_nonzero(param.grad)  # Count the number of non-zero gradient values.

    return num_grad_params.item()
if __name__ == '__main__':
    model = ResNet_base(BasicBlock_baseV2, layer_list=[2, 2, 2, 2], num_classes=4)
    print(model)
    tensor = torch.ones((1, 3, 64, 512))
    ys = model(tensor)
    ys = ys.sum()
    ys.backward(torch.ones_like(ys))
    print(count_grad_parameters(model))
    #print(model(tensor))