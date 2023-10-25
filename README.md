# Sample usage
Run an experiment with a specified dataset and algorithm:

`python main.py +wandb.name=example wandb.mode=online experiment=classification dataset=cifar10 algorithm=classifier`

If the dataset is found in cache, it will be loaded from there. Otherwise, it will be generated and cached.

supercloud command:
`python submit_job.py --name test --num_gpus 4 --num_cpus 20`
