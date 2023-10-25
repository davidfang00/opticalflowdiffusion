import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def is_square_of_two(num):
    if num <= 0:
        return False
    return num & (num - 1) == 0


class CnnEncoder(nn.Module):
    """
    Simple CNN encoder, fully customizable
    """
    def __init__(self,
                 input_shape=(3, 64, 64),
                 embedding_shape=(8, 16, 16),
                 activation_function='relu',
                 init_channels=16,
                 ):
        super().__init__()

        assert len(input_shape) == 3, "input_shape must be a tuple of length 3"
        assert len(embedding_shape) == 3, "embedding_shape must be a tuple of length 3"
        assert input_shape[1] == input_shape[2] and is_square_of_two(input_shape[1]), "input_shape must be square"
        assert embedding_shape[1] == embedding_shape[2], "embedding_shape must be square"
        assert input_shape[1] % embedding_shape[1] == 0, "input_shape must be divisible by embedding_shape"
        assert is_square_of_two(init_channels), "init_channels must be a square of 2"

        depth = int(math.sqrt(input_shape[1] / embedding_shape[1])) + 1
        channels_per_layer = [init_channels * (2 ** i) for i in range(depth)]
        self.act_fn = getattr(F, activation_function)

        self.downs = nn.ModuleList([])
        self.downs.append(nn.Conv2d(input_shape[0], channels_per_layer[0], kernel_size=3, stride=1, padding=1))

        for i in range(1, depth):
            self.downs.append(nn.Conv2d(channels_per_layer[i-1], channels_per_layer[i],
                                        kernel_size=3, stride=2, padding=1))

        # Bottleneck layer
        self.downs.append(nn.Conv2d(channels_per_layer[-1], embedding_shape[0], kernel_size=1, stride=1, padding=0))

    def forward(self, observation):
        hidden = observation
        for layer in self.downs:
            hidden = self.act_fn(layer(hidden))
        return hidden


class CnnDecoder(nn.Module):
    """
    Simple CNN decoder, fully customizable
    """
    def __init__(self,
                 embedding_shape=(8, 16, 16),
                 output_shape=(3, 64, 64),
                 activation_function='relu',
                 init_channels=16,
                 ):
        super().__init__()

        assert len(embedding_shape) == 3, "embedding_shape must be a tuple of length 3"
        assert len(output_shape) == 3, "output_shape must be a tuple of length 3"
        assert output_shape[1] == output_shape[2] and is_square_of_two(output_shape[1]), "output_shape must be square"
        assert embedding_shape[1] == embedding_shape[2], "input_shape must be square"
        assert output_shape[1] % embedding_shape[1] == 0, "output_shape must be divisible by input_shape"
        assert is_square_of_two(init_channels), "init_channels must be a square of 2"

        depth = int(math.sqrt(output_shape[1] / embedding_shape[1])) + 1
        channels_per_layer = [init_channels * (2 ** i) for i in range(depth)]
        self.act_fn = getattr(F, activation_function)

        self.ups = nn.ModuleList([])
        self.ups.append(nn.ConvTranspose2d(embedding_shape[0], channels_per_layer[-1],
                                           kernel_size=1, stride=1, padding=0))

        for i in range(1, depth):
            self.ups.append(nn.ConvTranspose2d(channels_per_layer[-i], channels_per_layer[-i-1],
                                               kernel_size=3, stride=2, padding=1, output_padding=1))

        self.output_layer = nn.ConvTranspose2d(channels_per_layer[0], output_shape[0],
                                               kernel_size=3, stride=1, padding=1)

    def forward(self, embedding):
        hidden = embedding
        for layer in self.ups:
            hidden = self.act_fn(layer(hidden))

        return self.output_layer(hidden)
