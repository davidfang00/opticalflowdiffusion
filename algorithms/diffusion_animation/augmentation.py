import torch
import torchvision

import random

class Augmentor():
    def __init__(self):
        def f_apply(func, args, p):
            def func_(i):
                return torch.cat([func(*args)(ii) for ii in torch.chunk(i, 2, dim=1)], dim=1)

            return torchvision.transforms.RandomApply([func_], p=p)

        def color_jitter(p):
            lim = 0.1
            params = [(random.random() - 0.5) * 2 * lim for _ in range(4)]
            base = [1, 1, 1, 0]
            params = [(b + r, b + r + 0.01) for b, r in zip(base, params)]
            return f_apply(torchvision.transforms.ColorJitter, params, p)

        def grayscale(p):
            return f_apply(torchvision.transforms.Grayscale, (3,), p)

        def gaussian_blur(p):
            sigma = random.random() * 0.5
            return f_apply(torchvision.transforms.GaussianBlur, (3, sigma), p)

        image_augs = torchvision.transforms.Compose([
            color_jitter(0.4),
            grayscale(0.1),
            gaussian_blur(0.2)
        ])

        def x(batch):
            batch = torchvision.transforms.RandomHorizontalFlip(p=1.0)(batch)
            batch[:, -1] = -1 * batch[:, -1]
            return batch

        def y(batch):
            batch = torchvision.transforms.RandomVerticalFlip(p=1.0)(batch)
            batch[:, -2] = -1 * batch[:, -2]
            return batch

        def rr_crop(batch):
            image_size = batch.shape[-1]
            params = torchvision.transforms.RandomResizedCrop.get_params(batch, [0.8, 1.0], [0.9, 1.1])
            batch[:, -2:] = batch[:, -2:] / image_size * torch.Tensor(params[-2:])[None, :, None, None].to(
                batch.device)
            return torchvision.transforms.functional.resized_crop(batch, *params, (image_size, image_size),
                                                                  antialias=False)

        whole_augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([x], p=0.3),
            torchvision.transforms.RandomApply([y], p=0.3),
            torchvision.transforms.RandomApply([rr_crop], p=0.15)
        ])

        def itemize(f):
            def itemized_f(t):
                chunked = torch.chunk(t, t.shape[0], dim=0)
                apply = [f(tt) for tt in chunked]
                return torch.cat(apply, dim=0)

            return itemized_f

        self.image_augs = itemize(image_augs)
        self.whole_augs = itemize(whole_augs)

    def __call__(self, batch):
        img, tgt, flow = batch
        aug_results = self.image_augs(torch.cat((img, tgt), dim=1))
        img, tgt = torch.chunk(aug_results, 2, dim=1)

        batch = torch.cat((img, tgt, flow), dim=1)
        batch = self.whole_augs(batch)

        return batch[:, :3], batch[:, 3:6], batch[:, 6:]