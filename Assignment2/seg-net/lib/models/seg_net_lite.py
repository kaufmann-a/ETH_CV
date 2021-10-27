from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filtersize, conv_padding):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(filtersize, filtersize), padding=conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filtersize, conv_padding):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(filtersize, filtersize), padding=conv_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        in_channels_down = [input_size] + down_filter_sizes[0:len(down_filter_sizes)-1]
        blocks_conv_down = [DownBlock(*params) for params in zip(in_channels_down, down_filter_sizes, kernel_sizes,
                                                                 conv_paddings)]
        layers_pooling = [nn.MaxPool2d(kernel_size=params[0], stride=params[1], return_indices=True) for params in
                          zip(pooling_kernel_sizes, pooling_strides)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.blocks_conv_down = nn.ModuleList(blocks_conv_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        in_channels_up = [down_filter_sizes[len(down_filter_sizes)-1]] + up_filter_sizes[0:len(up_filter_sizes)-1]
        blocks_conv_up = [UpBlock(*params) for params in zip(in_channels_up, up_filter_sizes, kernel_sizes,
                                                             conv_paddings)]
        layers_unpooling = [nn.MaxUnpool2d(kernel_size=params[0], stride=params[1]) for params in
                          zip(pooling_kernel_sizes, pooling_strides)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.blocks_conv_up = nn.ModuleList(blocks_conv_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.output_layer = nn.Conv2d(in_channels=up_filter_sizes[len(up_filter_sizes)-1], out_channels=11,
                                      kernel_size=(1, 1), padding=0)

    def forward(self, x):
        indices_array = []
        for idx, down in enumerate(self.blocks_conv_down):
            x = down(x)
            x, indices = self.layers_pooling[idx](x)
            indices_array.append(indices)
        for idx, up in enumerate(self.blocks_conv_up):
            x = self.layers_unpooling[idx](x, indices_array[len(self.blocks_conv_up)-idx-1])
            x = up(x)
        x = self.output_layer(x)
        return x

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
