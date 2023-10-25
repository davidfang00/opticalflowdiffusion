import torch
import torch.nn as nn
from torch.nn import functional as F


class CnnEncoder(nn.Module):
    """
    Simple cnn encoder that encodes a 64x64 image to embeddings
    """
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(1024, self.embedding_size)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, observation):
        batch_size = observation.shape[0]
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.fc(hidden.view(batch_size, 1024))
        return hidden


class CnnDecoder(nn.Module):
    """
    Simple Cnn decoder that decodes an embedding to 64x64 images
    """
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(embedding_size, 128)
        self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, embedding):
        batch_size = embedding.shape[0]
        hidden = self.fc(embedding)
        hidden = hidden.view(batch_size, 128, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


