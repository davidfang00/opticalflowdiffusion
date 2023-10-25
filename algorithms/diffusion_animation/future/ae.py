import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,
                 dim,
                 in_dim=3,
                 latent_dim=16,
                 down_dim_mults=(1, 2, 4, 8),
                 up_dim_mults=(8, 4),
                 resnet_block_groups = 8):

        super().__init__()

        self.init_conv = nn.Conv2d(in_dim, dim, 7, padding = 3)
        down_dims = [dim, *map(lambda m: dim * m, down_dim_mults)]
        down_dims = list(zip(down_dim_mults[:-1], down_dim_mults[1:]))
        up_dims = [*map(lambda m: dim * m, up_dim_mults), dim * up_dim_mults[-1]]
        up_dims = list(zip(up_dim_mults[:-1], up_dim_mults[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(down_dims):
            is_last = ind >= len(down_dims) - 1

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=None),
                block_klass(dim_in, dim_in, time_emb_dim=None),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = down_dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        for ind, (dim_in, dim_out) in enumerate(up_dims):
            is_last = ind == len(up_dims) - 1

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=None),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=None),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.out_dim = latent_dim
        self.final_res_block = block_klass(up_dims[-1] * 2, dim, time_emb_dim=None)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x)

            x = torch.cat([x, h.pop()], dim=1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x)
        x = self.final_conv(x)


class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 latent_dim=16,
                 out_dim=3,
                 down_dim_mults=(4, 8),
                 up_dim_mults=(8, 4, 2, 1),
                 resnet_block_groups = 8):
        super().__init__()

        self.init_conv = nn.Conv2d(latent_dim, dim, 1)
        down_dims = [*map(lambda m: dim * m, down_dim_mults), dim * down_dim_mults[-1]]

# TODO....

