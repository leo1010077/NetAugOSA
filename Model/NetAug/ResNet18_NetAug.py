from Model.base.ResNet_base import ResNet_base, ResNet
import torch.nn as nn
from typing import List
from Model.base.layers import (
    InvertedBlock,
    OpSequential,
    ResidualBlock,
    SeInvertedBlock,
    BasicBlock_base,
)
from Model.NetAug.layers import (
    DynamicConvLayer,
    DynamicConvLayer_padding,
    DynamicDsConvLayer,
    DynamicInvertedBlock,
    DynamicLinearLayer,
    DynamicSeInvertedBlock,
    DynamicBasicBlock_base,
)
from Model.NetAug.utils import aug_width, sync_width
from util.misc import make_divisible, torch_random_choices

class NetAugResNet18(ResNet):
    def __init__(
        self,
        base_net: ResNet,
        ResBlock: nn.Module,
        layer_list: list,
        aug_width_mult_list: List[float],
        n_classes: int,
        dropout_rate=0.0,
    ):
        nn.Module.__init__(self)
        max_width_mult = max(aug_width_mult_list)
        # input block
        base_input_block = base_net.backbone["input_block"]
        netaug_input_block = OpSequential(
            [
                DynamicConvLayer_padding(
                    3,
                    aug_width(
                        base_input_block.op_list[0].out_channels, aug_width_mult_list, 1
                    ),
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    use_bias=False
                )
            ]
        )
        # block1
        base_block1 = base_net.backbone["block"][0]
        #print(base_block1.op_list[0].conv1.conv.in_channels)
        netaug_block1 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block1.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block1.op_list[0].conv2.out_channels, aug_width_mult_list, 1))
        # block2
        base_block2 = base_net.backbone["block"][1]
        netaug_block2 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block2.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block2.op_list[0].conv2.out_channels, aug_width_mult_list, 1))
        # block3
        base_block3 = base_net.backbone["block"][2]
        netaug_block3 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block3.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block3.op_list[0].conv2.out_channels, aug_width_mult_list, 1))
        # block4
        base_block4 = base_net.backbone["block"][3]
        netaug_block4 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block4.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block4.op_list[0].conv2.out_channels, aug_width_mult_list, 1))
        # FChead
        base_FChead = base_net.head
        netaug_head = OpSequential(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                DynamicLinearLayer(
                    make_divisible(
                        base_FChead.op_list[-1].in_features * max_width_mult, 1
                    ),
                    n_classes,
                    dropout_rate=dropout_rate,
                ),
            ]
        )

        self.backbone = nn.ModuleDict(
            {
                "input_block": netaug_input_block,
                "block": nn.ModuleList([netaug_block1, netaug_block2, netaug_block3, netaug_block4]),
            }
        )
        self.head = netaug_head

    def _make_layer(self, ResBlock, blocks, in_channels, out_channel, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channel, downsample=True, stride=stride))
        self.in_channels = max(out_channel) * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, out_channel))

        return OpSequential(layers)

    @property
    def all_blocks(self):
        all_blocks = []
        print(self.backbone["block"])
        for stage in self.backbone["block"]:
            for block in stage.op_list:
                all_blocks.append(block)
        return all_blocks


    def set_active(self, mode: str, sync=False, generator=None):
        # input stem
        conv1= self.backbone["input_block"].op_list[0]
        if mode in ["min", "min_w"]:
            conv1.conv.active_out_channels = min(conv1.out_channels_list)
        elif mode in ["random", "min_e"]:
            conv1.conv.active_out_channels = torch_random_choices(
                conv1.out_channels_list,
                generator,
            )
        else:
            raise NotImplementedError
        if sync:
            conv1.conv.active_out_channels = sync_width(
                conv1.conv.active_out_channels
            )

        # stages
        in_channels = conv1.conv.active_out_channels
        for block in self.all_blocks:
            #print(block)
            if block.downsample is None:
                if mode in ["min", "min_w"]:
                    active_out_channels = min(block.conv1.out_channels_list)
                elif mode in ["random", "min_e"]:
                    active_out_channels = torch_random_choices(
                        block.conv1.out_channels_list,
                        generator,
                    )
                else:
                    raise NotImplementedError
                #print('down')

                if sync:
                    active_out_channels = sync_width(active_out_channels)

                block.conv1.conv.active_out_channels = active_out_channels
                block.conv2.conv.active_out_channels = active_out_channels

            else: # 有downsmple 每個block的第一個
                if mode in ["min", "min_w"]:
                    active_out_channels = min(block.conv1.out_channels_list)
                elif mode in ["random", "min_e"]:
                    active_out_channels = torch_random_choices(
                        block.conv1.out_channels_list,
                        generator,
                    )
                else:
                    raise NotImplementedError
                #print('down')

                if sync:
                    active_out_channels = sync_width(active_out_channels)

                block.conv1.conv.active_out_channels = active_out_channels
                block.conv2.conv.active_out_channels = active_out_channels
                block.downsample.conv.active_out_channels = active_out_channels

    def export(self) -> ResNet_base:
        export_model = ResNet_base.__new__(ResNet_base)
        nn.Module.__init__(export_model)
        # input_block
        input_block = OpSequential(
            [
                self.backbone["input_block"].op_list[0].export()
            ]
        )

        # stages
        stages = []



        # head
        head = OpSequential(
            [
                self.head.op_list[0].export(),
                self.head.op_list[1],
                self.head.op_list[2].export(),
                self.head.op_list[3].export(),
            ]
        )
        export_model.backbone = nn.ModuleDict(
            {
                "input_stem": input_stem,
                "stages": nn.ModuleList(stages),
            }
        )
        export_model.head = head
        return export_model

if __name__ == '__main__':
    print('start')
    basemodel = ResNet_base(BasicBlock_base, layer_list=[2, 2, 2, 2], num_classes=4)
    model = NetAugResNet18(base_net=basemodel, ResBlock=DynamicBasicBlock_base, layer_list=[2, 2, 2, 2], aug_width_mult_list=[2], n_classes=4)
    model.export()
    print(model)

