import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterToConv(nn.Module):
    def __init__(self, radius, out_dim=216):
        super(FilterToConv, self).__init__()
        self.radius = radius
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2, padding=2) # 17 -> 9
        self.conv2 = nn.Conv2d(8, 32, 5, stride=2, padding=2) # 9 -> 5
        self.conv3 = nn.Conv2d(32, out_dim // 9, 3, stride=2, padding=1) # 5 -> 3

        self.lin = nn.Linear(self.radius ** 2, out_dim)

    def forward(self, x):   # input (N, r2, i, i)
        return x
        bsz, imsize = x.shape[0], x.shape[-1]
        x = x.permute((0, 2, 3, 1)).reshape((-1, 1, self.radius, self.radius))  # (N*i*i, 1, r, r)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))   # (N*i*i, out_dim // 9, 3, 3)
        x = x.reshape((bsz, imsize, imsize, -1)).permute((0, 3, 1, 2))
        return x

class ConvToFilter(nn.Module):
    def __init__(self, radius, in_dim=81):
        super(ConvToFilter, self).__init__()
        self.radius = radius
        self.in_dim = in_dim
        self.conv1 = nn.ConvTranspose2d(in_dim // 9, 32, 3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 8, 5, stride=2, padding=2)
        self.conv3 = nn.ConvTranspose2d(8, 1, 5, stride=2, padding=2)
        self.final = nn.ConvTranspose2d(289, self.radius ** 2, 1)

        #self.lin = nn.Linear(in_dim, self.radius ** 2)

    def forward(self, x):   # input (N, in_dim, i, i)
        bsz, imsize = x.shape[0], x.shape[-1]
        x = x.permute((0, 2, 3, 1)).reshape((-1, self.in_dim // 9, 3, 3))   # (N*i*i, in_dim // 9, 3, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))   # (N*i*i, 1, r, r)
        x = x.reshape((bsz, imsize, imsize, 289)).permute((0, 3, 1, 2))  # (N, r2, i, i)
        x = self.final(x)
        return x
