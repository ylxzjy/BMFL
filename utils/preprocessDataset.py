# save to: like mnist_no_2_data.npy and mnist_no_2_lable.npy
# Divide data and labels to each client (getdataset)
import numpy as np
import struct
from PIL import Image
import os
import numpy as np
import imageio
import random
from tqdm import tqdm
import cv2

def preprocessData(args):
    getpath = './data'
    savepath = './cutdata'
    train_save_file = savepath+'/{}_{}_{}_clients/train'.format(args.dataset,args.epsilon,args.unsample_num)
    test_save_file = savepath+'/{}_{}_clients/test'.format(args.dataset,args.epsilon)
    if not os.path.exists(train_save_file):
        os.makedirs(train_save_file)
    if not os.path.exists(test_save_file):
        os.makedirs(test_save_file)
    print(train_save_file)
    print(test_save_file)
    if args.dataset == 'mnist':
        # train sataset
        print("Current working directory:", os.getcwd())
        train_data_file = getpath + '/mnist/train-images.idx3-ubyte'
        train_label_file = getpath + '/mnist/train-labels.idx1-ubyte'
        pretraindata(args,train_data_file,train_label_file,train_save_file)
        # test sataset
        test_data_file = getpath + '/mnist/t10k-images.idx3-ubyte'
        test_label_file = getpath + '/mnist/t10k-labels.idx1-ubyte'
        pretestdata(args,test_data_file,test_label_file,test_save_file)
    elif args.dataset == '8fashion':
        # train sataset
        print("Current working directory:", os.getcwd())
        train_data_file = getpath + '/8fashion/train-images-idx3-ubyte'
        train_label_file = getpath + '/8fashion/train-labels-idx1-ubyte'
        pretraindata(args,train_data_file,train_label_file,train_save_file)
        # test sataset
        test_data_file = getpath + '/8fashion/t10k-images-idx3-ubyte'
        test_label_file = getpath + '/8fashion/t10k-labels-idx1-ubyte'
        pretestdata(args,test_data_file,test_label_file,test_save_file)
    elif args.dataset == 'cifar10':
        # train sataset
        train_data_file = getpath + '/cifar10/train/data.npy'
        train_label_file = getpath + '/cifar10/train/label.npy'
        predatacifar10(args,train_data_file,train_label_file,train_save_file)
        # test sataset
        test_data_file = getpath + '/cifar10/test/data.npy'
        test_label_file = getpath + '/cifar10/test/label.npy'
        predatacifar10(args,test_data_file,test_label_file,test_save_file)  



# preprocess
def pretraindata(args,data_file,label_file,save_file):

    data = []
    # It's 47040016B, but we should set to 47040000B
    data_file_size = 47040016
    data_file_size = str(data_file_size - 16) + 'B'
    data_buf = open(data_file, 'rb').read()
    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)
    
    # It's 60008B, but we should set to 60000B
    label_file_size = 60008
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        
        pil_image_gray = img.convert('L')

        numpy_array = np.array(pil_image_gray)

        mean=0
        eps = args.epsilon
        resultImg = gauss_noise(numpy_array, mean, eps)

        label = labels[ii]
        data.append([np.array(resultImg), label])

    X_train = np.array([i[0] for i in data])
    y_train = np.array([i[1] for i in data])
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    if args.dataset == "8fashion":
        X_train, y_train = delete46(X_train,y_train)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    if args.unsample_num>0 and 'train' in save_file:
        print("select unsample_classes")
        np.random.seed(42)  
        unsampled_labels = np.random.choice(args.num_classes, args.unsample_num, replace=False)
        print("Unsampled labels:", unsampled_labels)

        indices_to_remove = np.where(np.isin(y_train, unsampled_labels))[0]

        num_to_remove = int(len(indices_to_remove) * 0.1)
        indices_to_remove = np.random.choice(indices_to_remove, num_to_remove, replace=False)

        indices_to_keep = np.where(~np.isin(y_train, unsampled_labels))[0]

        all_indices_to_keep = np.concatenate((indices_to_keep, indices_to_remove))

        X_train = X_train[all_indices_to_keep]
        y_train = y_train[all_indices_to_keep]

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

    np.save(save_file + '/data.npy', X_train)
    np.save(save_file + '/label.npy', y_train)


# precess
def pretestdata(args,data_file,label_file,save_file):

    data = []
    # It's 7840016B, but we should set to 7840000B
    data_file_size = 7840016
    data_file_size = str(data_file_size - 16) + 'B'
    data_buf = open(data_file, 'rb').read()
    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)

    # It's 10008B, but we should set to 10000B
    label_file_size = 10008
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)


    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        
        pil_image_gray = img.convert('L')

        numpy_array = np.array(pil_image_gray)

        mean=0
        eps = args.epsilon
        resultImg = gauss_noise(numpy_array, mean, eps)

        label = labels[ii]
        data.append([np.array(resultImg), label])

    X_train = np.array([i[0] for i in data])
    y_train = np.array([i[1] for i in data])
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    if args.dataset == "8fashion":
        X_train, y_train = delete46(X_train,y_train)
    np.save(save_file + '/data.npy', X_train)
    np.save(save_file + '/label.npy', y_train)

#preprocess cifar10
def predatacifar10(args,data_file,label_file,save_file):
    data = []
    X_train = np.load(data_file)
    y_train = np.load(label_file)
    print(X_train.shape)
    print(X_train.size)
    print(y_train[0])
    for ii in range(X_train.shape[0]):
        numpy_array = X_train[ii]
        
        mean=0
        eps = args.epsilon
        resultImg = gausscifar10_noise(numpy_array, mean, eps)

        label = y_train[ii]
        data.append([np.array(resultImg), label])

    if args.unsample_num>0 and 'train' in save_file:
        print("select unsample_classes")
        
        np.random.seed(42)  
        unsampled_labels = np.random.choice(args.num_classes, args.unsample_num, replace=False)
        print("Unsampled labels:", unsampled_labels)

        
        indices_to_remove = np.where(np.isin(y_train, unsampled_labels))[0]

        num_to_remove = int(len(indices_to_remove) * 0.1)
        indices_to_remove = np.random.choice(indices_to_remove, num_to_remove, replace=False)

       
        indices_to_keep = np.where(~np.isin(y_train, unsampled_labels))[0]

        
        all_indices_to_keep = np.concatenate((indices_to_keep, indices_to_remove))

        X_train = X_train[all_indices_to_keep]
        y_train = y_train[all_indices_to_keep]

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

    np.save(save_file + '/data.npy', X_train)
    np.save(save_file + '/label.npy', y_train)


def delete46(X_train,y_train):
    
    filtered_indices = np.where((y_train != 4) & (y_train != 6))
    X_train_filtered = X_train[filtered_indices]
    y_train_filtered = y_train[filtered_indices]

    y_train_filtered = np.where(y_train_filtered == 5, 4, y_train_filtered)
    y_train_filtered = np.where(y_train_filtered > 6, y_train_filtered - 2, y_train_filtered)

    return X_train_filtered, y_train_filtered


def gauss_noise(img, mean, eps):
    if eps!=0:
        img = cv2.GaussianBlur(img, (11, 11), -1)
        image = np.array(img / 255, dtype=float)
        print(image.shape)
        print(image.size)
        m, n = image.shape
        noise = np.random.laplace(mean, 1.0 / eps, image.size)
        noise.resize((m, n))
        out = image + noise
        resultImg = np.clip(out, 0.0, 1.0)
        resultImg = np.uint8(resultImg * 255.0)
        dst = cv2.GaussianBlur(resultImg, (11, 11), -1)
        return resultImg
    else:
        dst = cv2.GaussianBlur(img, (11, 11), -1)
        return dst

def gausscifar10_noise(img, mean, eps):
    if eps!=0:
        img = cv2.GaussianBlur(img, (11, 11), -1)
        image = np.array(img / 255, dtype=float)
        print(image.shape)
        print(image.size)
        k, m, n = image.shape
        noise = np.random.laplace(mean, 1.0 / eps, image.size)
        noise.resize((k, m, n))
        out = image + noise
        resultImg = np.clip(out, 0.0, 1.0)
        resultImg = np.uint8(resultImg * 255.0)
        dst = cv2.GaussianBlur(resultImg, (11, 11), -1)
        return resultImg
    else:
        dst = cv2.GaussianBlur(img, (11, 11), -1)
        return dst
