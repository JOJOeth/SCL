import torch.optim
import numpy as np
from torchvision import transforms as T
import os
import torch
from netCDF4 import Dataset
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random

kernel_3x3 = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])
kernel_5x5 = np.array([[1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1]])
def myAverage(input,kernel):
    return torch.tensor(ImgConvolve(input,kernel)*(1.0/np.sum(kernel)))
def Sobel(img, style):
    Gx = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    Gy = np.array([[-1,-2,-1],
                      [ 0, 0, 0],
                      [ 1, 2, 1]])
    sobelX = ImgConvolve(img,Gx)
    sobelY = ImgConvolve(img,Gy)
    if(style == 0):
        return sobelX
    if(style == 1):
        return sobelY
    if(style == 2):
        return abs(sobelX)+abs(sobelY)
def Prewitt(img,style):
    Gx = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
    Gy = np.array([[-1,-1,-1],
                      [ 0, 0, 0],
                      [ 1, 1, 1]])
    prewittX = ImgConvolve(img,Gx)
    prewittY = ImgConvolve(img,Gy)
    prewittAdd = np.zeros_like(prewittX)
    [rows,cols] = prewittAdd.shape
    for i in range(0,rows):
        for j in range(0,cols):
            prewittAdd[i][j] = prewittX[i][j]+prewittY[i][j]
    if(style == 0):
        return prewittX
    if(style == 1):
        return prewittY
    if(style == 2):
        return prewittAdd
def ImgConvolve(input,kernel):
    input = input[0]
    WImg = input.shape[0]
    HImg = input.shape[1]
    Wkernel = kernel.shape[0]
    Hkernel = kernel.shape[1]
    AddW = (Wkernel-1)/2
    AddH = (Hkernel-1)/2
    ImgTemp = np.zeros([WImg + int(AddW*2),HImg + int(AddH*2)])
    ImgTemp[int(AddW):int(AddW)+WImg,int(AddH):int(AddH)+HImg] = input[:,:]
    output = np.zeros_like(a=ImgTemp)
    for i in range(int(AddW),int(AddW)+WImg):
        for j in range(int(AddH),int(AddH)+HImg):
            output[i][j] = int(np.sum(ImgTemp[i-int(AddW):i+int(AddW)+1,j-int(AddW):j+int(AddW)+1]*kernel))#计算平均值
    return output[int(AddW):int(AddW)+WImg,int(AddH):int(AddH)+HImg]
class AddPepperNoise(object):
    def __init__(self, snr, p, ma, mi):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p
        self.ma = ma
        self.mi = mi
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            c, h, w = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(1, h, w),
                                    p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=0)

            img_[mask == 1] = self.ma  # 盐噪声
            img_[mask == 2] = self.mi  # 椒噪声
            return torch.tensor(img_)
        else:
            return torch.tensor(img)
class SobelFiltering(object):
    def __init__(self, mode):
        self.mode = mode
    def __call__(self, mat):
        s = torch.tensor(Sobel(mat, 0))
        return torch.tensor(torch.stack([s, s, s], axis=0))
class Average(object):
    def __init__(self, kernel):
        self.kernel = kernel
    def __call__(self, mat):
        s = torch.tensor(ImgConvolve(mat,self.kernel)*(1.0/np.sum(self.kernel)))
        return torch.stack([s, s, s], axis=0)
class PrewittFiltering(object):
    def __init__(self, mode):
        self.mode = mode
    def __call__(self, mat):
        s = torch.tensor(Prewitt(mat, 0))
        return torch.stack([s, s, s], axis=0)
class AddRandomNoise(object):
    def __init__(self, snr, p, ma, mi):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p
        self.ma = ma
        self.mi = mi
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            c, h, w = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(1, h, w),
                                    p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=0)

            img_[mask == 1] = self.mi  # 盐噪声
            img_[mask == 2] = self.mi  # 椒噪声
            return torch.tensor(img_)
        else:
            return torch.tensor(img)
class Gaussian_noise(object):
    def __init__(self, ma, mi):
        self.ma = ma
        self.mi = mi
    def __call__(self, img):
        img_ = np.array(img).copy()
        # 产生高斯 noise
        noise = np.random.normal(0, (self.ma - self.mi)/10, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        return torch.tensor(gaussian_out)
class climatedataset():
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    def __getitem__(self, index):
        d1 = self.dataset1[index]
        d2 = self.dataset2[index]
        return d1, d2
    def __len__(self):
        return len(self.dataset1)
def env(path):
    list = os.listdir(path)
    dt_list = []
    for i in list:
        filename = path + '/' + i
        dt = Dataset(filename)
        u = torch.from_numpy(dt['u'][:, 0, :, :])
        v = torch.from_numpy(dt['v'][:, 0, :, :])
        r = torch.from_numpy(dt['u'][:, 0, :, :])
        t = torch.from_numpy(dt['v'][:, 2, :, :])
        z = torch.from_numpy(dt['z'][:, 0, :, :])
        w = torch.from_numpy(dt['t'][:, 0, :, :])
        l = [u, v, r, t, z, w]
        dt = torch.stack(l, dim=1)
        dt_list.append(dt)
    dts = torch.cat(dt_list, dim=0)
    return dts
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
pre_o = pre("E:/ERA5/W/WW/train/pre")
def data_un(mix, augment, shift, num_classes, env_train_path, env_test_path, pre_train_path, pre_test_path):
    def contrast_augmented(sample, mean, std, ma, mi):
        a = []
        for i in range(sample.shape[0]):
            b = []
            for j in range(sample.shape[1]):
                mat = sample[i][j]
                mat = torch.stack([mat, mat, mat], axis=2).numpy()
                transforms = T.Compose([T.ToTensor(),
                                        T.Normalize([mean[j] for p in range(3)], [std[j] for q in range(3)]),
                                        T.Resize((100,200)),
                                        ])
                mat_aug = transforms(mat)[0, :, :]
                b.append(mat_aug)
            b = np.stack(b, axis=0)
            a.append(b)
        return np.stack(a, axis=0)
    def normalized(sample, mean, std):
        a = []
        for i in range(sample.shape[0]):
            b = []
            for j in range(sample.shape[1]):
                mat = sample[i][j]
                mat = torch.stack([mat, mat, mat], axis=2).numpy()
                transforms = T.Compose([T.ToTensor(),
                                        T.Normalize([mean[j] for p in range(3)], [std[j] for q in range(3)]),
                                        # T.Resize(100),
                                        # T.RandomCrop((37, 73)),
                                        ])
                mat_aug = transforms(mat)[0, :, :]
                b.append(mat_aug)
            b = np.stack(b, axis=0)
            a.append(b)
        return np.stack(a, axis=0)
    train_env = env(env_train_path)
    test_env = env(env_test_path)
    channels = train_env.shape[1]
    mean_train = [torch.mean(train_env[:, i, :, :]) for i in range(channels)]
    std_train = [torch.std(train_env[:, i, :, :]) for i in range(channels)]
    if mix == True:
        aug_data_train = normalized(train_env, mean_train, std_train)
        aug_data_test = normalized(test_env, mean_train, std_train)
        ma = torch.max(torch.tensor(aug_data_train))
        mi = torch.min(torch.tensor(aug_data_train))
        aug_data_trainn = contrast_augmented(train_env, mean_train, std_train, ma, mi)
        aug_data_testt = contrast_augmented(test_env, mean_train, std_train, ma, mi)
        if shift > 0:
            train_shifted = aug_data_trainn[
                            np.append(np.array([0 for i in range(shift)]), range(aug_data_train.shape[0] - shift)), :,
                            :, :]
            test_shifted = aug_data_testt[
                           np.append(np.array([0 for i in range(shift)]), range(aug_data_test.shape[0] - shift)), :, :,
                           :]
        else:
            shift = -1 * shift
            train_shifted = aug_data_train[
                            np.append(range(shift, aug_data_train.shape[0]), np.array([-1 for i in range(shift)])), :,
                            :, :]
            test_shifted = aug_data_test[
                           np.append(range(shift, aug_data_test.shape[0]), np.array([-1 for i in range(shift)])), :, :,
                           :]
        data_train = torch.utils.data.TensorDataset(torch.tensor(aug_data_train, dtype=float))
        data_train_shifted = torch.utils.data.TensorDataset(torch.tensor(train_shifted, dtype=float))
        data_test = torch.utils.data.TensorDataset(torch.tensor(aug_data_test, dtype=float))
        data_test_shifted = torch.utils.data.TensorDataset(torch.tensor(test_shifted, dtype=float))
        shift_dataset_train = climatedataset(data_train, data_train_shifted)
        shift_dataset_test = climatedataset(data_test, data_test_shifted)
        del data_train, data_train_shifted, data_test, data_test_shifted, train_env, test_env
        return shift_dataset_train, shift_dataset_test
    elif shift is not None:
        aug_data_train = normalized(train_env, mean_train, std_train)
        aug_data_test = normalized(test_env, mean_train, std_train)
        if shift > 0:
            train_shifted = aug_data_train[np.append(np.array([0 for i in range(shift)]), range(aug_data_train.shape[0] - shift)), :, :, :]
            test_shifted = aug_data_test[np.append(np.array([0 for i in range(shift)]), range(aug_data_test.shape[0] - shift)), :, :, :]
        else:
            shift = -1 * shift
            train_shifted = aug_data_train[np.append(range(shift, aug_data_train.shape[0]), np.array([-1 for i in range(shift)])), :, :, :]
            test_shifted = aug_data_test[np.append(range(shift, aug_data_test.shape[0]), np.array([-1 for i in range(shift)])), :, :, :]
        data_train = torch.utils.data.TensorDataset(torch.tensor(aug_data_train, dtype=float))
        data_train_shifted = torch.utils.data.TensorDataset(torch.tensor(train_shifted, dtype=float))
        data_test = torch.utils.data.TensorDataset(torch.tensor(aug_data_test, dtype=float))
        data_test_shifted = torch.utils.data.TensorDataset(torch.tensor(test_shifted, dtype=float))
        shift_dataset_train = climatedataset(data_train, data_train_shifted)
        shift_dataset_test = climatedataset(data_test, data_test_shifted)
        del data_train, data_train_shifted, data_test, data_test_shifted, train_env, test_env
        return shift_dataset_train, shift_dataset_test
    elif augment == True:
        aug_data_train0 = normalized(train_env, mean_train, std_train)
        aug_data_test0 = normalized(test_env, mean_train, std_train)
        ma = torch.max(torch.tensor(aug_data_train0))
        mi = torch.min(torch.tensor(aug_data_train0))
        aug_data_train = contrast_augmented(train_env, mean_train, std_train, ma, mi)
        aug_data_test = contrast_augmented(test_env, mean_train, std_train, ma, mi)
        data_train = torch.utils.data.TensorDataset(torch.tensor(aug_data_train0, dtype=float))
        data_test = torch.utils.data.TensorDataset(torch.tensor(aug_data_test0, dtype=float))
        data_train_aug = torch.utils.data.TensorDataset(torch.tensor(aug_data_train, dtype=float))
        data_test_aug = torch.utils.data.TensorDataset(torch.tensor(aug_data_test, dtype=float))
        aug_dataset_train =climatedataset (data_train_aug, data_train)
        aug_dataset_test = climatedataset(data_test_aug, data_test)
        del aug_data_train, aug_data_test, aug_data_train0, aug_data_test0, data_train, data_test
        return aug_dataset_train, aug_dataset_test
    else:
        aug_data_train = normalized(train_env, mean_train, std_train)
        aug_data_test = normalized(test_env, mean_train, std_train)
        data_train = torch.utils.data.TensorDataset(torch.tensor(aug_data_train, dtype=float))
        data_test = torch.utils.data.TensorDataset(torch.tensor(aug_data_test, dtype=float))
        dataset_train = climatedataset(data_train, data_train)
        dataset_test = climatedataset(data_test, data_test)
        del data_train, data_test, aug_data_train, aug_data_test
        return dataset_train, dataset_test
def data_li(augment, shift, num_classes, env_train_path, env_test_path, pre_train_path, pre_test_path):
    def normalized(sample, mean, std):
        a = []
        for i in range(sample.shape[0]):
            b = []
            for j in range(sample.shape[1]):
                mat = sample[i][j]
                mat = torch.stack([mat, mat, mat], axis=2).numpy()
                transforms = T.Compose([T.ToTensor(),
                                        T.Normalize([mean[j] for p in range(3)], [std[j] for q in range(3)]),
                                        T.Resize((100,200)),
                                        ])
                mat_aug = transforms(mat)[0, :, :]
                b.append(mat_aug)
            b = np.stack(b, axis=0)
            a.append(b)
        return np.stack(a, axis=0)
    train_env = env(env_train_path)
    test_env = env(env_test_path)
    def labels(num_classes, pre_train_path, pre_test_path):
        train_data = pre(pre_train_path)
        n = len(train_data)
        nn = len(pre_o)
        pres = []
        for i in pre_o:
            pres.append(i)
        press = []
        for i in train_data:
            press.append(i)
        pre_sorted = sorted(pres)
        pre_split = [0]
        for i in range(num_classes - 1):
            pre_split.append(pre_sorted[int(nn * (i + 1) / num_classes) - 1])
        labels_train = []
        for i in range(len(train_data)):
            label = 0
            for j in range(num_classes):
                if train_data[i] > pre_split[j]:
                    label = j
            labels_train.append(label)
        labels_test = []
        if pre_test_path is not None:
            test_pre = pre(pre_test_path)
            for i in range(len(test_pre)):
                label = 0
                for j in range(num_classes):
                    if test_pre[i] > pre_split[j]:
                        label = j
                labels_test.append(label)
        pre_prototypes = []
        for i in range(num_classes):
            x = 0
            m = 0
            for j in range(n):
                if labels_train[j] == i:
                    m += 1
                    x += press[j]
            if m != 0:
                x /= m
            pre_prototypes.append(x)
        return labels_train, labels_test, pre_prototypes
    channels = train_env.shape[1]
    mean_train = [torch.mean(train_env[:, i, :, :]) for i in range(channels)]
    std_train = [torch.std(train_env[:, i, :, :]) for i in range(channels)]
    targets_train, targets_test, pre_prototypes = labels(num_classes, pre_train_path, pre_test_path)
    aug_data_train = normalized(train_env, mean_train, std_train)
    aug_data_test = normalized(test_env, mean_train, std_train)
    data_train = torch.utils.data.TensorDataset(torch.tensor(aug_data_train, dtype=float), torch.tensor(np.array(targets_train), dtype=float))
    data_test = torch.utils.data.TensorDataset(torch.tensor(aug_data_test, dtype=float), torch.tensor(np.array(targets_test), dtype=float))
    del train_env, test_env, aug_data_train, aug_data_test
    return data_train, data_test, pre_prototypes, pre(pre_test_path)
