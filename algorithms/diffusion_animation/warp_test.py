import torch
import torchvision
from .warp import warp, nan_mse
import random
import math
from tqdm import tqdm

#for level in [1, 2, 4]:
for _ in tqdm(range(10000)):
    #for level in [1, 2, 4]:
    #for level in [1, 2, 4, 8]:
    #for level in range(1, 17):#[1, 2, 4, 8, 16, 32, 64]:
    for level in [2]:
        b, c, x, y = 1, 1, 128, 128
        src = torch.zeros((b, c, x, y)).cuda()
        """
        one_x = math.floor(random.random() * x)
        one_y = math.floor(random.random() * y)
        src[0, 0, one_x, one_y] = 1.0
        """
        """
        src = torch.Tensor([[[[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]]]]).cuda()
        flow = torch.Tensor([[[[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.5, 0.0]],
                             [[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]]]).cuda()
        src = src.uniform_()
        flow = 2 * flow.uniform_() - 1.0
        """
        flow = torch.zeros((b, 2, x, y)).cuda()
        """
        for _ in range(1):
            one_x = math.floor(random.random() * x)
            one_y = math.floor(random.random() * y)
            flow[0, 0, one_x, one_y] = round(random.random() * 4 - 2) / 2
        """
        flow = flow.uniform_()
        flow_1 = flow.clone()
        flow_1 = torch.round(4 * flow_1.uniform_() - 2.0)
        flow_2 = flow.clone()
        flow_2 = 4 * flow_2.uniform_() - 2.0
        flow = torch.where(flow < 0.5, flow_1, flow_2)
        src = src.uniform_()

        a = math.floor(random.random() * level)
        b = math.floor(random.random() * level)
        #a, b = 1, 0

        def method_a():
            single_warp = warp(src, None, flow, scale=level, set_nans=False, mode='forward', rep='flow', offset=[a, b]) / (level ** 2)
            return single_warp

        def method_b():
            high_warp = warp(src, None, flow, scale=1, set_nans=True, mode='forward', rep='flow', offset=[0, 0])
            low_warp = warp(high_warp, None, torch.zeros_like(flow), scale=level, set_nans=False, mode='forward', rep='flow', offset=[a, b]) / (level ** 2)
            return low_warp

        def eval():
            warped_gt = warp(src, None, flow, mode='forward', rep='flow')

            single_warp = method_a()
            double_warp = warp(warped_gt, None, torch.zeros_like(flow), mode='forward', rep='flow', scale=level, offset=[a, b]) / (level ** 2)
            #loss = nan_mse(double_warp, single_warp)
            diff = torch.abs(double_warp - single_warp)

            if torch.max(diff) > 1e-4:
                torchvision.utils.save_image(warp(warped_gt, None, torch.zeros_like(flow), mode='forward', rep='flow', scale=level, set_nans=False, offset=[a, b]), f'double_warped.png')
                torchvision.utils.save_image(warp(src, None, flow, scale=level, set_nans=False, mode='forward', rep='flow', offset=[a, b]), f'single_warped.png')
                print(src)
                print(flow)
                print(warped_gt)
                print(warp(warped_gt, None, torch.zeros_like(flow), mode='forward', rep='flow', scale=level, offset=[a, b]))
                print(warp(src, None, flow, scale=level, set_nans=False, mode='forward', rep='flow', offset=[a, b]))
                input(str(torch.max(diff)) + str(level) + str(a) + str(b) + str(torch.argmax(diff))) 

        def are_they_equal():
            warped_gt = warp(src, None, flow, mode='forward', rep='flow')
            double_warp = warp(warped_gt, None, torch.zeros_like(flow), mode='forward', rep='flow', scale=level, offset=[a, b]) / (level ** 2)

            matrices = []
            grad_matrices = []
            comp = torch.zeros_like(double_warp)
            comp = comp.uniform_()
            for method in [method_a(), method_b()]:
                matrices.append(method)
                with torch.set_grad_enabled(True):
                    method.requires_grad_(True)
                    #loss = self.loss(tgt_, cond, flow_, override_flow=p_flows / self.flow_max)
                    #loss = torch.nn.functional.mse_loss(method, double_warp)
                    loss = torch.nn.functional.mse_loss(method, comp)
                    loss.backward()
                    grad_matrices.append(-method.grad.clone()) # negative so this reflects the drcn of grad descent
                    method.grad = None
                    method.requires_grad_(False)

            if not torch.allclose(grad_matrices[0], grad_matrices[1]):
                print(torch.allclose(matrices[0], matrices[1]), torch.allclose(grad_matrices[0], grad_matrices[1]))
                print(torch.max(torch.abs(grad_matrices[0] - grad_matrices[1])), torch.argmax(torch.abs(grad_matrices[0] - grad_matrices[1])))
            #print(torch.allclose(matrices[0], matrices[1]), torch.allclose(grad_matrices[0], grad_matrices[1]))
            #print(grad_matrices[0] - grad_matrices[1])
        are_they_equal()


#torchvision.utils.save_image(torch.abs(self.model._warp(warped_gt, torch.zeros_like(flow_), scale=level, set_nans=False) - self.model._warp(cond, flow_pred, scale=level, set_nans=False)), f'levels_var2_{level}.png')
