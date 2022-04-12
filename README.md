# SCL
PyTorch implementation of SCL: Self-supervised Classification of Weather Systems based on Spatiotemporal Contrastive Learning

### Training ResNet encoder:
Simply run the following to pre-train a ResNet encoder using SimCLR on the CIFAR-10 dataset:
```
python main.py 
```

### Distributed Training
With distributed data parallel (DDP) training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

## What is SCL?
SCL is short for "Self-supervised Classification of Weather Systems based on Spatiotemporal Contrastive Learning". We propose self-supervised framework based on contrastive learning for representation learning on multivariate meteorological data, in which spatiotemporal transformations are applied for data augmentations to utilize the invariance of key features after transformations.

<p align="center">
  <img src="./pic/framework.png" width="1000"/>
</p>


## Usage
Simply run for single GPU or CPU training:
```
python main.py
```

For distributed training (DDP), use for every process in nodes, in which N is the GPU number you would like to dedicate the process to:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

`--nr` corresponds to the process number of the N nodes we make available for training.

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `40`).

```
python evaluation.py
```

or in place:
```
python evaluation.py --model_path=./save --epoch_num=40
```


## Configuration
The configuration of training can be found in: config/config.yaml. An example config.yaml file:
```
# data path
env_train_path: "your_train_data_path"
env_test_path: "your_test_data_path"
pre_train_path: "your_train_precipitation_path"
pre_test_path: "your_test_precipitation_path"

# distributed training
nodes: 1
gpus: 1 # I recommend always assigning 1 GPU to 1 node
nr: 0 # machine nr. in node (0 -- nodes - 1)
dataparallel: 0 # Use DataParallel instead of DistributedDataParallel
workers: 0
dataset_dir: "./datasets"

# train options
seed: 2 # sacred handles automatic seeding when passed in the config
batch_size: 256
start_epoch: 0
epochs: 50
dataset: "Climate"
pretrain: False
num_classes: 16
num_levels: 16
scl_checkpoint: "your_checkpoint_path"
logistic_checkpoint: "your_logistic_checkpoint_path"
optimizer_checkpoint: "your_optimizer_checkpoint_path"
scheduler_checkpoint: "your_scheduler_checkpoint_path"
resume: False


# model options
resnet: "resnet18"
projection_dim: 256 # "to project the representation to a 256-dimensional latent space"
scl_ps: "scl"
logistic_ps: "logistic"
optimizer_ps: "optimizer"
scheduler_ps: "scheduler"
mix: False
shift: 2
augment: False 

# loss options
optimizer: "Adam" # or LARS 
weight_decay: 1.0e-5 
temperature: 0.5

# reload options
model_path: "save" # set to the directory containing `checkpoint_##.tar`
epoch_num: 100 # set to checkpoint number
reload: False

# logistic regression options
logistic_batch_size: 256
logistic_epochs: 1200

# simulation
simulation_epochs: 1000

# evaluation
evaluation_mode: "cluster" # or linear
```

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```
