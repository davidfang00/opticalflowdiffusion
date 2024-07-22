# Optical Flow Estimation & Diffusion

This repo contains code for optical flow estimation from video and flow generation with diffusion models.

## Usage

Run an experiment with a specified dataset and algorithm:

`python main.py +wandb.name=example wandb.mode=online experiment=matrix_flow dataset=sintel algorithm=flow_diffuser`

If the dataset is found in cache, it will be loaded from there. Otherwise, it will be generated and cached.

supercloud command:
`python submit_job.py --name test --num_gpus 4 --num_cpus 20`
