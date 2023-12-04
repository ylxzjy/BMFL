# BMFL
BM-FL: A Balanced Weight Strategy for Multi-stage Federated Learning Against Multi-client Data Skewing

# Requirements

pytorch >= 1.6

torchvision >= 0.9.0

## model

BC-GAN

## dataset 
Dowload MNIST, fashion mnist and cifar10 dataset at the official website.
MNIST: download and extract the mnist. Then, place those four ubyte files to .\data\mnist
Homepage: http://yann.lecun.com/exdb/mnist/

8-fashion: download and extract the Fashion MNIST. Then, place the place those four ubyte files folder to .\datasets\8fashion
Homepage: https://www.kaggle.com/datasets/zalando-research/fashionmnist/

CIFAR-10: download and extract CIFAR-10 python version. Then, place the thumbnails128x128 folder to .\datasets\cifar10  
Homepage: https://www.cs.toronto.edu/~kriz/cifar.html



## Usage

```
python main.py --dataset mnist --num_classes 10 --channels 1 --img_size 28 --w_epochs 4 --train_ep 5 --epoch 500 --num_clients 100 --num_its 5
```

```
python main.py --dataset 8fashion --num_classes 8 --channels 1 --img_size 28 --w_epochs 4 --train_ep 5 --epoch 500 --num_clients 100 --num_its 5
```

```
python main.py --dataset cifar10 --num_classes 10 --channels 3 --img_size 32 --w_epochs 4 --train_ep 5 --epoch 500 --num_clients 100 --num_its 5
```
