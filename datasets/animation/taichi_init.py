import sys
from taichi import TaiChiDataset
from omegaconf import DictConfig, OmegaConf

cfg = OmegaConf.create({
	"image_size": 256,
	"scale_down": 1.0,		
	"calculate_flows": True,
	"frame_distance": 5,
	"flow_method": "raft",
        "flow_batch_size": 96
})

# Grab the arguments that are passed in
try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except:
    my_task_id = 0
    num_tasks = 0

device = "cuda:" + str(my_task_id % 2)
print(str(my_task_id) + ' out of ' + str(num_tasks))

ds = TaiChiDataset(cfg, split='training', device=device, mod=str(my_task_id) + "," + str(num_tasks))
ds = TaiChiDataset(cfg, split='test', device=device, mod=str(my_task_id) + "," + str(num_tasks))
