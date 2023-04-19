from Model.base.ResNet_baseV2 import ResNet_base
import torch.nn as nn
import torch
from typing import List
from Model.base.layers import (
    InvertedBlock,
    OpSequential,
    ResidualBlock,
    SeInvertedBlock,
    BasicBlock_base,
    BasicBlock_baseV2,
    PoolLayer,
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
from Model.NetAug.utils import aug_width, sync_width, sort_channels_resblock, sort_channels_resblock_BN
from util.misc import make_divisible, torch_random_choices

class NetAugResNet18(ResNet_base):
    def __init__(
        self,
        base_net: ResNet_base,
        ResBlock: nn.Module,
        layer_list: list,
        aug_width_mult_list: List[float],
        n_classes: int,
        dropout_rate=0.0,
    ):
        nn.Module.__init__(self)
        max_width_mult = max(aug_width_mult_list)
        self.ResBlock = ResBlock
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
                ),
            PoolLayer(kernel_size=3, stride=2, padding=1)
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
                                  aug_width(base_block2.op_list[0].conv2.out_channels, aug_width_mult_list, 1),
                                    stride=2)
        # block3
        base_block3 = base_net.backbone["block"][2]
        netaug_block3 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block3.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block3.op_list[0].conv2.out_channels, aug_width_mult_list, 1),
                                    stride=2)
        # block4
        base_block4 = base_net.backbone["block"][3]
        netaug_block4 = self._make_layer(ResBlock,
                                  layer_list[0],
                                  make_divisible(base_block4.op_list[0].conv1.conv.in_channels * max_width_mult, 1),
                                  aug_width(base_block4.op_list[0].conv2.out_channels, aug_width_mult_list, 1),
                                    stride=2)
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
        layers.append(ResBlock(conv1 = DynamicConvLayer_padding(in_channels, out_channel, kernel_size=3, padding=1, stride=stride, use_bias=False),
                               conv2 = DynamicConvLayer_padding(max(out_channel), out_channel, kernel_size=3, padding=1, stride=1, use_bias=False, act_func=None),
                               downsample = DynamicConvLayer_padding(in_channels, out_channel, kernel_size=1, padding=0, stride=stride, use_bias=False, act_func=None)))
        self.in_channels = max(out_channel) * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(
                conv1=DynamicConvLayer_padding(self.in_channels, out_channel, kernel_size=3, padding=1, stride=1,
                                        use_bias=False),
                conv2=DynamicConvLayer_padding(max(out_channel), out_channel, kernel_size=3, padding=1, stride=1,
                                        use_bias=False, act_func=None),
                downsample=None))

        return OpSequential(layers)


    def _make_layer_export(self, NetAug_block):
        layers = []
        #print(NetAug_block[0])
        layers.append(self.ResBlock(conv1 = NetAug_block[0].conv1.export(),
                               conv2 = NetAug_block[0].conv2.export(),
                               downsample = NetAug_block[0].downsample.export()))
        for i in range(len(NetAug_block) - 1):
            layers.append(self.ResBlock(
                conv1=NetAug_block[i+1].conv1.export(),
                conv2=NetAug_block[i+1].conv2.export(),
                downsample=None))

        return OpSequential(layers)


    @property
    def all_blocks(self):
        all_blocks = []
        #print(self.backbone["block"])
        for stage in self.backbone["block"]:
            for block in stage.op_list:
                all_blocks.append(block)
        return all_blocks

    def set_active(self, mode: str, sync=False, generator=None):
        # input stem
        conv1= self.backbone["input_block"].op_list[0]
        #print(conv1)
        if mode in ["min", "min_w"]:
            min_aug = min(conv1.out_channels_list)
            conv1.conv.active_out_channels = min_aug
        elif mode in ["random", "min_e"]:
            random_aug = torch_random_choices(conv1.out_channels_list,generator,)
            conv1.conv.active_out_channels = random_aug
        elif mode in ["max"]:
            random_aug = max(conv1.out_channels_list)
            conv1.conv.active_out_channels = random_aug
        else:
            raise NotImplementedError
        if sync:
            conv1.conv.active_out_channels = sync_width(conv1.conv.active_out_channels)

        # stages
        in_channels = conv1.conv.active_out_channels
        for block in self.all_blocks:
            #print(block)
            if block.downsample is None:

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
                elif mode in ["max"]:
                    active_out_channels = max(block.conv1.out_channels_list)
                else:
                    raise NotImplementedError
                #print('down')

                if sync:
                    active_out_channels = sync_width(active_out_channels)

                block.conv1.conv.active_out_channels = active_out_channels
                block.conv2.conv.active_out_channels = active_out_channels
                block.downsample.conv.active_out_channels = active_out_channels

        # head
        # head 應該不用調因為input是自動的 output是channel數
        #print(self.head.op_list[1])

    def export(self) -> ResNet_base:
        export_model = ResNet_base.__new__(ResNet_base)
        nn.Module.__init__(export_model)
        # input_block
        input_block = OpSequential(
            [
                self.backbone["input_block"].op_list[0].export(),
                PoolLayer(kernel_size=3, stride=2, padding=1)
            ]
        )

        # stages
        stages = []
        for block in self.backbone['block']:
            #print(block)
            stages.append(self._make_layer_export(block.op_list))


        # head
        #print(self.head)
        head = OpSequential(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                self.head.op_list[1].export()
            ]
        )
        export_model.backbone = nn.ModuleDict(
            {
                "input_block": input_block,
                "block": nn.ModuleList(stages),
            }
        )
        export_model.head = head
        return export_model

    def sort_channels(self) -> None:
        for block in self.all_blocks:

            #print(block)
            sort_channels_resblock(block.conv1)
            sort_channels_resblock(block.conv2)
            if block.downsample !=None:
                sort_channels_resblock(block.downsample)
    def sort_channels_BN(self) -> None:
        for block in self.all_blocks:

            #print(block)
            sort_channels_resblock_BN(block.conv1)
            sort_channels_resblock_BN(block.conv2)
            if block.downsample !=None:
                sort_channels_resblock_BN(block.downsample)

if __name__ == '__main__':
    print('start')
    tensor = torch.zeros((1, 3, 64, 512))
    basemodel = ResNet_base(BasicBlock_baseV2, layer_list=[2, 2, 2, 2], num_classes=4)
    model = NetAugResNet18(base_net=basemodel, ResBlock=BasicBlock_baseV2, layer_list=[2, 2, 2, 2],
                           aug_width_mult_list=[1, 2], n_classes=4)
    model.sort_channels_BN()

    # model2 = model.export()
    # print(model2(tensor))
    #model.module.set_active('min')


