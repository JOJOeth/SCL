import os
import numpy as np
import torch
import torchvision
import argparse


# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SCL
from scl import SCL
from scl.modules import NT_Xent, get_resnet
from scl.modules.transformations import TransformsSCL
from scl.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook
from datamanger import data_un


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = data[0][0].type(torch.FloatTensor).cuda(non_blocking=True)
        x_j = data[1][0].type(torch.FloatTensor).cuda(non_blocking=True)
        del data
        # positive pair, with encoding
        z_i, z_j = model(x_i, x_j)
        del x_i, x_j
        loss = criterion(z_i, z_j)
        del z_i, z_j
        loss.backward()
        optimizer.step()


        # if args.nr == 0 and step % 1 == 0:
        #     print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch
def test(args, train_loader, model, criterion, writer):
    loss_epoch = 0
    for step, data in enumerate(train_loader):
        model.zero_grad()
        x_i = data[0][0].type(torch.FloatTensor).cuda(non_blocking=True)
        x_j = data[1][0].type(torch.FloatTensor).cuda(non_blocking=True)
        z_i, z_j = model(x_i, x_j)
        del x_i, x_j
        loss = criterion(z_i, z_j)

        # if args.nr == 0 and step % 1 == 0:
        #     print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")


        loss_epoch += loss.item()
    return loss_epoch
def main(args):
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False

    train_dataset, test_dataset = data_un(args.mix, args.augment, args.shift, args.num_classes,
                                        args.env_train_path,
                                        args.env_test_path,
                                        args.pre_train_path,
                                        args.pre_test_path
                                )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    # n_features = 512 * 7 * 7
    # initialize model
    model = SCL(True, encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)


    model = model.to(args.device)

    writer = SummaryWriter(log_dir='testlog', flush_secs=10)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch_train = train(args, train_loader, model, criterion, optimizer, writer)
        if epoch % 1 == 0:
            loss_epoch_test = test(args, test_loader, model, criterion, writer)
            writer.add_scalar("Loss/test", loss_epoch_test / len(test_loader), epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch_test / len(test_loader)}\t lr: {round(lr, 5)}"
            )
        if scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 1 == 0:
            save_model(args, model, optimizer, scheduler)


        writer.add_scalar("Loss/train", loss_epoch_train / len(train_loader), epoch)

        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch_train / len(train_loader)}\t lr: {round(lr, 5)}"
        )
        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SCL")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    main(args)
