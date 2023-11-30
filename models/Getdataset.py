import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset,SubsetRandomSampler
from torchvision import datasets, transforms
from utils.config import args_parser
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import struct
import random


class GetDataSet(object):
    def __init__(self, args):
        self.args = args
        self.test_loader = None
        self.NICO_train_loader = []
        self.NICO_train_dataset = []

        rootpath = './cutdata'
        train_data = rootpath + '/{}_{}_{}_clients/train/data.npy'.format(self.args.dataset,self.args.epsilon,self.args.unsample_num)
        train_label = rootpath + '/{}_{}_{}_clients/train/label.npy'.format(self.args.dataset,self.args.epsilon,self.args.unsample_num)


        test_data = rootpath + '/{}_{}_clients/test/data.npy'.format(self.args.dataset,self.args.epsilon)
        test_label = rootpath + '/{}_{}_clients/test/label.npy'.format(self.args.dataset,self.args.epsilon)

        if self.args.dataset == 'mnist' or self.args.dataset == '8fashion':
            
            #分配给num_clients个train_loader
            test_images = np.load(test_data)
            test_labels = np.load(test_label)

            X_test = torch.from_numpy(test_images.reshape(-1, 1, 28, 28)).float()  # 输入 x 张量

            Y_test = torch.from_numpy(test_labels).long()  # 输入 y 张量
            X_test = X_test / 255
            X_test = transforms.Normalize([0.5], [0.5])(X_test)
            print(X_test.size())
            print(Y_test.size())
            testDataset = torch.utils.data.TensorDataset(X_test, Y_test)  # 合并训练数据和目标数据
            print("testDataset size:", len(testDataset))
            self.test_loader = torch.utils.data.DataLoader(
                dataset=testDataset,
                batch_size=self.args.train_bs,
                shuffle=True,
                num_workers=1  # set multi-work num read data
            )

            train_images = np.load(train_data)
            train_labels = np.load(train_label)

            X_train = torch.from_numpy(train_images.reshape(-1, 1, 28, 28)).float()  # 输入 x 张量

            Y_train = torch.from_numpy(train_labels).long()  # 输入 y 张量
            X_train = X_train / 255
            X_train = transforms.Normalize([0.5], [0.5])(X_train)
            print(X_train.size())
            print(Y_train.size())
            self.classdivide(X_train,Y_train)
        elif self.args.dataset == 'cifar10':
            #分配给num_clients个train_loader
            test_images = np.load(test_data)
            test_labels = np.load(test_label)

            X_test = np.transpose(test_images, (0, 3, 1, 2))  # 输入 x 张量
            X_test = torch.from_numpy(X_test).float()
            Y_test = torch.from_numpy(test_labels).long()  # 输入 y 张量
            X_test = X_test / 255
            X_test = transforms.Normalize((0.5,),(0.5,))(X_test)
            print(X_test.size())
            print(Y_test.size())
            testDataset = torch.utils.data.TensorDataset(X_test, Y_test)  # 合并训练数据和目标数据
            print("testDataset size:", len(testDataset))
            self.test_loader = torch.utils.data.DataLoader(
                dataset=testDataset,
                batch_size=self.args.train_bs,
                shuffle=True,
                num_workers=1  # set multi-work num read data
            )

            train_images = np.load(train_data)
            train_labels = np.load(train_label)

            X_train = np.transpose(train_images, (0, 3, 1, 2))  # 输入 x 张量
            X_train = torch.from_numpy(X_train).float()

            Y_train = torch.from_numpy(train_labels).long()  # 输入 y 张量
            X_train = X_train / 255
            X_train = transforms.Normalize((0.5,),(0.5,))(X_train)
            print(X_train.size())
            print(Y_train.size())
            self.classdivide(X_train,Y_train)

        else:
            exit("Error: Getdataset ")

    def classdivide(self,X_train,Y_train):
        for i in range(self.args.num_classes):
            # 找到 Y_train 中标签为 2 的索引
            indices = (Y_train == i).nonzero().squeeze()

            # 使用这些索引从 X_train 中提取相应的数据
            selected_X = X_train[indices]
            selected_Y = Y_train[indices]
            print(selected_X.shape,selected_Y.shape)
            trainDataset = torch.utils.data.TensorDataset(selected_X, selected_Y)  # 合并训练数据和目标数据
            self.divide_data(trainDataset,i)
        
        print(len(self.NICO_train_dataset))
        for i in range(self.args.num_clients):
            print("len NICO_train_dataset[]: ", i, len(self.NICO_train_dataset[i]))
            if len(self.NICO_train_dataset[i]) == 0:
                print("len==0")
                for j in range(self.args.num_clients):
                    if len(self.NICO_train_dataset[j])!=0:
                        # 使用 copy.deepcopy() 创建数据集副本
                        import copy
                        self.NICO_train_dataset[i] = copy.deepcopy(self.NICO_train_dataset[j])
                        print("len NICO_train_dataset[]: ", i, len(self.NICO_train_dataset[i]))
                        break
            train_data_loader = torch.utils.data.DataLoader(
                dataset=self.NICO_train_dataset[i],
                batch_size=self.args.train_bs,
                shuffle=True,
                num_workers=1  # set multi-work num read data
            )
            self.NICO_train_loader.append(train_data_loader)
        del self.NICO_train_dataset

    def divide_data(self, trainDataset, j):
        #使用随机种子，将总的数据集分配给所有的客户端
        random.seed(42)  # 设置随机种子
        numbers = [random.random() for _ in range(self.args.num_clients)]
        total_sum = sum(numbers)
        normalized_numbers = [x / total_sum for x in numbers]
        scaled_numbers = [int(x * len(trainDataset)) for x in normalized_numbers]
        start_idx=0
        end_idx=0
        index=0
        print("label ",j ," to client")
        for i in scaled_numbers:
            end_idx = end_idx + i
            print(j,start_idx,end_idx)
            subset_sampler = torch.utils.data.Subset(trainDataset, range(start_idx,end_idx))
            if j==0:
                self.NICO_train_dataset.append(subset_sampler)
            else:
                self.NICO_train_dataset[index]=torch.utils.data.ConcatDataset([self.NICO_train_dataset[index], subset_sampler])
            start_idx = start_idx + i
            index=index+1
        print("*"*20)

    

if __name__ == '__main__':
    args = args_parser()
    c = GetDataSet(args=args)
