import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scl import SCLL
from scl.modules import LogisticRegression, get_resnet
from scl.modules.transformations import TransformsSCL
from model import load_optimizer, save_model
from utils import yaml_config_hook
from datamanger import data_li

from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta
import pandas as pd


def pre(path):
    list = os.listdir(path)
    df = None
    for i in list:
        filename = path + '/' + i
        data = Dataset(filename)
        time = [datetime(1900, 1, 1, 0, 0, 0) + timedelta(hours=int(i) - 3) for i in data['time'][:]]
        latitude = data['latitude'][:]
        longitude = data['longitude'][:]
        data = pd.DataFrame(data['tp'][:].flatten(), index=pd.MultiIndex.from_product([time, latitude, longitude]))
        data = data.groupby(level=[0]).mean().resample('H').sum()
        data = data
        if df is None:
            df = data
        else:
            df = pd.concat([df, data], axis=0)
    train_df = df * 1e3
    fore = 1
    m = np.array(train_df[0])
    return m[np.append(np.array([0 for i in range(fore)]), range(train_df.shape[0] - fore))]

def inference(loader, scl_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        with torch.no_grad():
            h, z = scl_model(x, x)

        feature_vector.extend(z.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(scl_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, scl_model, device)
    test_X, test_y = inference(test_loader, scl_model, device)
    return train_X, train_y, test_X, test_y

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, test_pre, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test = validation_dataset(test, test_pre)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True
    )
    return train_loader, test_loader

def train_linear(args, loader, model, criterion, optimizer, epoch):
    loss_epoch = 0
    predicts = []
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.type(torch.LongTensor).to(args.device)

        output = model(x).to(args.device)
        loss = criterion(output, y)

        predicted = output.argmax(1)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        predicts.append(predicted)
    predicts = torch.cat(predicts, dim=0)
    if epoch % 200 == 0:
        np.save('labels', predicts.cpu())

    return loss_epoch

class validation_dataset():
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    def __getitem__(self, index):
        d1 = self.dataset1[index]
        d2 = self.dataset2[index]
        return d1, d2

    def __len__(self):
        return len(self.dataset1)

def test_linear(args, loader, model, criterion, pre_prototypes, epoch):
    loss_epoch = 0
    dis1_epoch = 0
    Sb_epoch = 0
    Ss_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        l = len(y)
        model.zero_grad()
        y = np.array(y)
        data = x[0].to(args.device)
        targets = x[1].type(torch.LongTensor).to(args.device)
        output = model(data).to(args.device)
        loss = criterion(output, targets)

        predicted = output.argmax(1)
        if epoch % 200 == 0 and epoch != 0:
            for i in range(args.num_classes):
                Sb_epoch += (list(predicted).count(i) / len(predicted) - list(targets).count(i)/len(predicted))**2 /args.num_classes
                Ss_epoch += min(list(predicted).count(i)/ len(predicted), list(targets).count(i)/len(predicted))
        pre_predicted = [pre_prototypes[predicted[i]] for i in range(y.shape[0])]
        pre_sorted = sorted(pre_predicted)
        y_sorted  = sorted(y)
        m = np.sum(pre_predicted)
        n = np.sum(y)
        pre_predicted = np.array(pre_predicted)
        dis1_epoch += np.sum(np.abs(pre_predicted - y))
        loss_epoch += loss.item()
    return loss_epoch, dis1_epoch, Sb_epoch, Ss_epoch, pre_sorted, y_sorted

def cluster(num_classes, num_levels, train_embeddings, test_embeddings, pre_train_path, pre_test_path):
    train_embeddings = torch.tensor(train_embeddings).cuda()
    test_embeddings = torch.tensor(test_embeddings).cuda()
    kmeans = KMeans(n_clusters=num_classes, mode='euclidean', verbose=1)
    labels_train = kmeans.fit_predict(train_embeddings)
    np.save('labels', labels_train.cpu())
    def prototypes_contrast(train_embeddings):
        index_list = []
        pro_list = []
        for i in range(num_classes):
            m = []
            n = []
            for j in range(labels_train.shape[0]):
                if labels_train[j] == i:
                    m.append(j)
                    n.append(train_embeddings[j])
            index_list.append(m)  # 存放每种类型对应的下标
            pro_list.append(n)  # 存放每种类型对应的env
        env_prototypes = []
        proj_dim = train_embeddings.shape[1]
        for i in range(num_classes):
            x = torch.zeros(proj_dim).cuda()
            for sample in pro_list[i]:
                x += sample
            x /= len(pro_list[i])
            env_prototypes.append(x)  # 得到env的prototypes
        train_pre = pre(pre_train_path)
        pre_prototypes = []
        for i in range(num_classes):
            x = 0
            for index in index_list[i]:
                x += train_pre[index]
            x /= len(index_list[i])
            pre_prototypes.append(x)  # prototypes的平均降雨
        return pre_prototypes, env_prototypes, index_list
    def levels(num_levels, pre_train_path, pre_test_path):
        train_data = pre(pre_train_path)
        n = len(train_data)
        pres = []
        for i in train_data:
            pres.append(i)
        pre_sorted = sorted(pres)
        pre_split = [0]
        for i in range(num_levels - 1):
            pre_split.append(pre_sorted[int(n * (i + 1) / num_levels) - 1])
        levels_train = []
        for i in range(len(train_data)):
            label = 0
            for j in range(num_levels):
                if train_data[i] > pre_split[j]:
                    label = j
            levels_train.append(label)
        levels_test = []
        if pre_test_path is not None:
            test_pre = pre(pre_test_path)
            for i in range(len(test_pre)):
                label = 0
                for j in range(num_levels):
                    if test_pre[i] > pre_split[j]:
                        label = j
                levels_test.append(label)
        return levels_train, levels_test, pre_split
    def eu_dis(a, b):
        return torch.sqrt(torch.sum(torch.pow(a - b, 2)))
    def labels_contrast(test_embeddings, env_prototypes_contrast):
        labels = []
        for i in range(len(test_embeddings)):
            label = 0
            d = eu_dis(env_prototypes_contrast[0], test_embeddings[i])
            for j in range(len(env_prototypes_contrast)):
                dis = eu_dis(env_prototypes_contrast[j], test_embeddings[i])
                if dis < d:
                    d = dis
                    label = j
            labels.append(label)
        return labels

    pre_prototypes, env_prototypes_contrast, index_list = prototypes_contrast(train_embeddings)
    labels_test = labels_contrast(test_embeddings, env_prototypes_contrast)
    test_pre = pre(pre_test_path)

    return labels_test, test_pre, pre_prototypes

def test_cluster(epochs, num_levels, labels, pre_prototypes):
    l = len(labels)
    dis1 = 0
    Sb = 0
    Ss = 0
    pre_test = pre(args.pre_test_path)
    def levels(num_levels, ps, pre_test_path):
        train_data = ps
        n = len(train_data)
        pres = []
        for i in train_data:
            pres.append(i)
        pre_sorted = sorted(pres)
        pre_split = [0]
        for i in range(num_levels - 1):
            pre_split.append(pre_sorted[int(n * (i + 1) / num_levels) - 1])
        levels_train = []
        for i in range(len(train_data)):
            label = 0
            for j in range(num_levels):
                if train_data[i] > pre_split[j]:
                    label = j
            levels_train.append(label)
        levels_test = []
        if pre_test_path is not None:
            test_pre = pre(pre_test_path)
            for i in range(len(ps)):
                label = 0
                for j in range(num_levels):
                    if test_pre[i] > pre_split[j]:
                        label = j
                levels_test.append(label)
        return levels_train, levels_test
    ps = []
    for i in range(epochs):
        print(i)
        a = 0
        b = 0
        c = 0
        d = 0
        outputs = []
        for j in range(len(labels)):
            p = pre_prototypes[labels[j]]
            dis1 += np.abs(p - pre_test[j])
            ps.append(p)
        levels_output, levels_test = levels(num_levels, ps, args.pre_test_path)
        for i in range(num_levels):
            Sb += (levels_output.count(i) / l - levels_test.count(i) / l) ** 2
            Ss += min(levels_output.count(i) / l, levels_test.count(i) / l)
    dis1 /= (epochs * l)
    Sb /= (num_levels * epochs)
    Ss /= epochs
    return dis1, Sb, Ss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCL")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False


    train_dataset, test_dataset, pre_prototypes, test_pre = data_li(False, None, args.num_classes,
                                            args.env_train_path,
                                            args.env_test_path,
                                            args.pre_train_path,
                                            args.pre_test_path
                                            )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    scl_model = SCLL(encoder, args.projection_dim, n_features)
    scl_model_fp = os.path.join(args.model_path, args.scl_checkpoint)
    scl_model.load_state_dict(torch.load(scl_model_fp, map_location=args.device.type))
    scl_model = scl_model.to(args.device)
    scl_model.eval()

    logistic_model = LogisticRegression(scl_model.n_features, args.num_classes)

    if args.resume:
        logistic_model_fp = os.path.join(args.model_path, args.logistic_checkpoint)
        logistic_model.load_state_dict(torch.load(logistic_model_fp, map_location=args.device.type))
    logistic_model = logistic_model.to(args.device)
    logistic_model.eval()

    optimizer, scheduler = load_optimizer(args, logistic_model)
    if args.resume:
        optimizer_fp = os.path.join(args.model_path, args.optimizer_checkpoint)
        optimizer.load_state_dict(torch.load(optimizer_fp, map_location=args.device.type))
        scheduler_fp = os.path.join(args.model_path, args.scheduler_checkpoint)
        scheduler.load_state_dict(torch.load(scheduler_fp, map_location=args.device.type))

    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")

    if args.evaluation_mode == 'cluster':
        (train_X, train_y, test_X, test_y) = get_features(
            scl_model, train_loader, test_loader, args.device
        )

        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, test_pre, args.logistic_batch_size
        )
        train_embeddings = train_X
        test_embeddings = test_X
        labels, test_pre, pre_prototypes = cluster(args.num_classes, args.num_levels, train_embeddings, test_embeddings, args.pre_train_path, args.pre_test_path)
        dis1, Sb, Ss = test_cluster(args.simulation_epochs, args.num_levels, labels, pre_prototypes)
    elif args.evaluation_mode == 'linear':
        (train_X, train_y, test_X, test_y) = get_features(
            scl_model, train_loader, test_loader, args.device
        )
        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, test_pre, args.logistic_batch_size
        )
        for epoch in range(args.logistic_epochs):
            loss_epoch_train = train_linear(
                args, arr_train_loader, logistic_model, criterion, optimizer, epoch
            )
            if epoch % 200 == 0 and epoch != 0:
                loss_epoch_test, dis1_epoch, Sb_epoch, Ss_epoch, \
                pre_sorted, y = test_linear(
                    args, arr_test_loader, logistic_model, criterion, pre_prototypes, epoch
                )
                if epoch % 200 == 0 and epoch != 0:
                    l =  len(pre_sorted)
                    myprob = []
                    itsprob =[]
                    writer = SummaryWriter(log_dir='testlog', flush_secs=10)
                    writer.add_scalar('JOJO/dis1', dis1_epoch / len(arr_test_loader), epoch)
                    writer.add_scalar('JOJO/Sb', Sb_epoch / len(arr_test_loader), epoch)
                    writer.add_scalar('JOJO/Ss', Ss_epoch / len(arr_test_loader), epoch)
                    writer.add_scalar('JOJO/loss', loss_epoch_train / len(arr_train_loader), epoch)
                    print(
                        f"Epoch [{epoch}/{args.logistic_epochs}]\t "
                        f"Loss_train: {loss_epoch_train / len(arr_train_loader)}\n "
                        f"Epoch [{epoch}/{args.logistic_epochs}]\t "
                        f"Loss_test: {loss_epoch_test / len(arr_test_loader)}\n "
                        f"Epoch [{epoch}/{args.logistic_epochs}]\t "
                        f"dis1: {dis1_epoch / len(arr_test_loader)}\n "
                        f"Epoch [{epoch}/{args.logistic_epochs}]\t "
                        f"Sb: {Sb_epoch/ len(arr_test_loader)}"
                        f"Epoch [{epoch}/{args.logistic_epochs}]\t "
                        f"Ss: {Ss_epoch/ len(arr_test_loader)}"
                    )
                else:
                    writer = SummaryWriter(log_dir='testlog', flush_secs=10)
                    writer.add_scalar('JOJO/dis1', dis1_epoch / len(arr_test_loader), epoch)
                    writer.add_scalar('JOJO/loss', loss_epoch_train / len(arr_train_loader), epoch)






