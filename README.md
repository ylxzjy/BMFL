# BMFL
BM-FL: A Balanced Weight Strategy for Multi-stage Federated Learning Against Multi-client Data Skewing

# Requirements

pytorch >= 1.6

torchvision >= 0.9.0

## model

BC-GAN

## dataset 

mnist, 8fashion, cifar10


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
