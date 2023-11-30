import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from models.Nets import *
import os
from models.Getdataset import GetDataSet
from torch.nn import DataParallel
import logging
import math
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self, id, training_set):
        self.id = id
        self.training_set = training_set  # 客户端训练集
        self.weights = [0.8]  # 客户端权重,初始权重0.8,下一次权重更新，使用滑动权重策略

class TrustedServer:
    def __init__(self, clients):
        self.clients = clients  # 客户端列表
        self.accuracy_array = []  # 准确率数组,每一次聚合前记录一次本地ITS准确率
        self.local_index=0


class ClientUpdate(object):
    def __init__(self, args):
        self.args = args
        self.auxiliary_loss = nn.CrossEntropyLoss().to(device)#真假损失函数
        self.adversarial_loss = nn.BCELoss().to(device)#分类损失函数
        self.cdataloader=[]
        

    def get_accifar10model(self):
        global_dnet, global_gnet, clients_gnet, clients_dnet = None, None, None, None
        if self.args.model == 'BC-GAN' and self.args.dataset == 'cifar10':
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            #全局判别器
            global_dnet = Discriminatorcifar(self.args).to(device)
            global_gnet = Generatorcifar(self.args).to(device)
            #中间信任服务器的生成器和判别器
            clients_gnet = [Generatorcifar(self.args).to(device) for _ in range(self.args.num_its)]
            clients_dnet = [Discriminatorcifar(self.args).to(device) for _ in range(self.args.num_its)]
        else:
            exit("Error: Clients -- get_model() --- no model")

        global_dnet.apply(weights_init_normal)
        global_gnet.apply(weights_init_normal)
        for model in clients_gnet:
            model.apply(weights_init_normal)
        for model in clients_dnet:
            model.apply(weights_init_normal)

        optimizerg = [torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2)) for model in clients_gnet]
        optimizerd = [torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2)) for model in clients_dnet]
        return global_dnet, global_gnet, clients_gnet, clients_dnet, optimizerg, optimizerd


    def get_acmnistmodel(self):
        global_dnet, global_gnet, clients_gnet, clients_dnet = None, None, None, None
        if self.args.model == 'BC-GAN':
            print('-----model:{}   dataset:{}'.format(self.args.model, self.args.dataset))
            #全局判别器
            global_dnet = Discriminatormnist(self.args).to(device)
            global_gnet = Generatormnist(self.args).to(device)
            #中间信任服务器的生成器和判别器
            clients_gnet = [Generatormnist(self.args).to(device) for _ in range(self.args.num_its)]
            clients_dnet = [Discriminatormnist(self.args).to(device) for _ in range(self.args.num_its)]
        else:
            exit("Error: Clients -- get_model() --- no model")

        global_dnet.apply(weights_init_normal)
        global_gnet.apply(weights_init_normal)
        for model in clients_gnet:
            model.apply(weights_init_normal)
        for model in clients_dnet:
            model.apply(weights_init_normal)

        optimizerg = [torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2)) for model in clients_gnet]
        optimizerd = [torch.optim.Adam(model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2)) for model in clients_dnet]
        return global_dnet, global_gnet, clients_gnet, clients_dnet, optimizerg, optimizerd

    #BM_FL中单ITS上的训练mnist和fashion
    def client_train_bmfl(self, generator, discriminator,optimizer_G, optimizer_D, discriminatorlast, trustedServer):
        epoch_loss=[]
        for wep in range(self.args.w_epochs):
            #获取当前客户端权重下的数据集
            hdataloader = self.getdataloader(trustedServer)
            for lep in range(self.args.train_ep):
                for i, (imgs, labels) in enumerate(hdataloader):
                    batch_size = imgs.shape[0]
                    # Adversarial ground truths
                    valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
                    fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(device)

                    # Configure input
                    real_imgs = imgs.to(device)
                    labels = labels.to(device)
                    # -----------------
                    #  Train Generator
                    # ----------------- 
                    optimizer_G.zero_grad()
                    # Sample noise and labels as generator input
                    #z服从标准正态分布（均值为0，标准差为1）的随机数
                    z = torch.randn((batch_size, self.args.latent_dim)).to(device)
                    gen_labels = torch.randint(0, self.args.num_classes, (batch_size,)).to(device)

                    # Generate a batch of images
                    gen_imgs = generator(z, gen_labels)

                    # Loss measures generator's ability to fool the discriminator
                    gen_imgs = gen_imgs.to(device)
                    validity, pred_label = discriminator(gen_imgs)
                    g_loss = 0.5 * (self.adversarial_loss(validity, valid) * 1 + self.auxiliary_loss(pred_label, gen_labels) * 1)

                    g_loss.backward()
                    optimizer_G.step()
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    optimizer_D.zero_grad()

                    # Loss for real images
                    real_pred, real_aux = discriminator(real_imgs)
                    d_real_loss = (self.adversarial_loss(real_pred, valid) * 0.5 + self.auxiliary_loss(real_aux,
                                                                                             labels) * 1.5) / 2

                    # Loss for fake images
                    fake_pred, fake_aux = discriminator(gen_imgs.detach())
                    d_fake_loss = (self.adversarial_loss(fake_pred, fake) * 1.6 + self.auxiliary_loss(fake_aux,
                                                                                            gen_labels) * 0.4) / (2)

                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()
                    epoch_loss.append(d_loss)
                    #d_loss = d_real_loss
                    # Calculate discriminator accuracy
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    ##pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)
                    ##gt = np.concatenate([labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)
                    torch.cuda.empty_cache()
                print("[epoch %d/%d] [WEpoch %d/%d] [LEpoch %d/%d] [D loss: %f , acc: %f%%] [G loss: %f]" % (trustedServer.local_index,self.args.epoch,wep,self.args.w_epochs,lep,self.args.train_ep, d_loss.item(), 100 * d_acc, g_loss.item()))

                    
            # 使用上一次权重更新的判别器模型更新数据集权重
            closs = []
            avgloss = 0.0
            for j in range(len(self.cdataloader)):
                # 计算每个客户端数据集的平均损失以及整体的平均损失
                temploss = 0.0
                for i, (imgs, labels) in enumerate(self.cdataloader[j]):
                    with torch.no_grad():
                        batch_size = imgs.shape[0]
                        # Configure input
                        real_imgs = imgs.to(device)
                        labels = labels.to(device)
                        # Loss for real images
                        real_pred, real_aux = discriminatorlast(real_imgs)
                        d_real_loss = self.auxiliary_loss(real_aux, labels)
                        temploss = temploss + d_real_loss
                    torch.cuda.empty_cache()
                temploss = temploss / len(self.cdataloader[j])
                closs.append(temploss)
            avgloss = sum(closs)/len(closs)
            #print(avgloss,closs)
            for j in range(len(self.cdataloader)):
                # 更新权重
                cwight = trustedServer.clients[j].weights[-1]
                cwight = cwight + (closs[j] - avgloss) * (trustedServer.local_index +1) * 0.3
                trustedServer.clients[j].weights.append(cwight)
                #滑动更新
                self.sliding_average_update(trustedServer.clients[j].weights,self.args.slidL)
                if trustedServer.clients[j].weights[-1] > self.args.wup:
                    trustedServer.clients[j].weights[-1] = self.args.wup
                if trustedServer.clients[j].weights[-1] < self.args.wlow:
                    trustedServer.clients[j].weights[-1] = self.args.wlow

            discriminatorlast.load_state_dict(discriminator.state_dict())
        trustedServer.local_index = trustedServer.local_index +1
        return sum(epoch_loss)/len(epoch_loss)

    # 预训练
    def client_train_preacgan_mnist(self, generator, discriminator,optimizer_G, optimizer_D, train_loader):
        for epoch in range(self.args.pre_epochs):
            for i, (imgs, labels) in enumerate(train_loader):
                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
                fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(device)

                # Configure input
                real_imgs = imgs.to(device)
                labels = labels.to(device)
                

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.randn((batch_size, self.args.latent_dim)).to(device)
                gen_labels = torch.randint(0, self.args.num_classes, (batch_size,)).to(device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)
                # print(gen_imgs[0])
                # print("gen_imgs--------")
                # Loss measures generator's ability to fool the discriminator
                gen_imgs = gen_imgs.to(device)
                validity, pred_label = discriminator(gen_imgs)
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) * 0.5 + self.auxiliary_loss(pred_label, gen_labels) * 1.5)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                real_imgs = real_imgs.to(device)
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
        self.sample_mnistimage(self.args.num_classes, epoch + 1, generator)

    def sample_mnistimage(self, n_row, batches_done, generator):
        """Saves a grid of generated digits ranging from 0 to num_classes"""
        # Sample noise
        z = torch.rand((n_row ** 2, self.args.latent_dim)).to(device)
        # Get labels ranging from 0 to num_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        # labels = np.array([random.randint(0, 9) for _ in range(10) for _ in range(10)])
        labels = torch.LongTensor(labels).to(device)
        gen_imgs = generator(z, labels)
        file_path = './generate/{}/{}_{}_{}clients_{}itss/'.format(self.args.dataset,self.args.epsilon,self.args.unsample_num,self.args.num_clients,self.args.num_its)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        save_image(gen_imgs.data, file_path+"%d.png" % batches_done, nrow=n_row, normalize=True)


    def getdataloader(self,trustedServer):
        self.cdataloader.clear()
        merged_train_dataset = trustedServer.clients[0].training_set.dataset
        idx=0
        for i in trustedServer.clients:
            X_train = []
            Y_train = []
            for sample in i.training_set.dataset:
                # print(sample[0])  # 输出输入特征，检查其内容
                # print(sample[1])  # 输出标签，检查其内容
                X_train.append(sample[0])
                Y_train.append(sample[1])
            
            # 将 X_train 和 Y_train 转换为张量
            X_train = torch.stack(X_train)
            Y_train = torch.stack(Y_train)

            #print(len(X_train), i.weights[-1], math.floor(len(X_train) * i.weights[-1]))
            # 取前len*w行数据
            X_train = X_train[:math.floor(len(X_train) * i.weights[-1])]
            Y_train = Y_train[:math.floor(len(Y_train) * i.weights[-1])]

            trainDataset = torch.utils.data.TensorDataset(X_train, Y_train)  # 合并训练数据和目标数据
            dataloader = torch.utils.data.DataLoader(
                dataset=trainDataset,
                batch_size=self.args.train_bs,
                shuffle=True,
                num_workers=4  # set multi-work num read data
            )
            
            if idx == 0:
                merged_train_dataset = trainDataset
            else:
                merged_train_dataset = torch.utils.data.ConcatDataset([merged_train_dataset, trainDataset])
            
            idx=3
            self.cdataloader.append(dataloader)
            del X_train
            del Y_train
            del trainDataset
            del dataloader

        dataloader = torch.utils.data.DataLoader(
            dataset=merged_train_dataset,
            batch_size=self.args.train_bs,
            shuffle=True,
            num_workers=4  # set multi-work num read data
        )
        del merged_train_dataset
        # print(len(dataloader))
        return dataloader

    def FedAvg(self, global_net, clients_net):
        globel_dict = global_net.state_dict()
        for k in globel_dict.keys():
            globel_dict[k] = torch.stack([clients_net[i].state_dict()[k].float() for i in range(len(clients_net))], 0).mean(0)
        global_net.load_state_dict(globel_dict)

    def updata_model(self, global_net, clients_net):
        for model in clients_net:
            model.load_state_dict(global_net.state_dict())

    def test(self, global_net, test_loader):
        global_net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for index, (data, target) in enumerate(test_loader):
                batch_size = data.shape[0]
                valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
                data, target = data.to(device), target.to(device)
                real_pred, real_aux = global_net(data)

                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, target)) / 2
                total_loss += d_real_loss.item()  # 累积损失
                pred = real_aux.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()  # 累积正确的样本数
                total_samples += batch_size  # 累积总样本数
        average_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples
        return average_loss, accuracy


    #滑动平均
    def sliding_average_update(self, arr, L):
        # 获取数组的长度
        length = len(arr)

        if length < L:
            avg = sum(arr) / length
        else:
            avg = sum(arr[-L:]) / L

        arr[-1] = avg

    # 预训练cifar10
    def client_train_preacgan_cifar(self, generator, discriminator,optimizer_G, optimizer_D, train_loader):
        for epoch in range(self.args.pre_epochs):
            for i, (imgs, labels) in enumerate(train_loader):
                batch_size = self.args.train_bs  ## 64
                N_Class = self.args.num_classes # 10
                img_size = self.args.img_size  # 32
                # Adversarial ground truths
                valid = torch.FloatTensor(batch_size).fill_(1.0).to(device)
                fake = torch.FloatTensor(batch_size).fill_(0.0).to(device)
                print(device,"*"*20)
                # Configure input
                real_imgs = imgs.to(device)  # 64 3 32 32
                labels = labels.to(device)

                if i == len(train_loader)-2:
                    break
                real_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, N_Class, 1, 1).contiguous() # 64 10 1 1
                real_y = real_y.expand(-1, -1, img_size, img_size) # 64 10 32 32

                # Sample noise and labels as generator input
                noise = torch.randn(batch_size, self.args.latent_dim,1,1).to(device) # 64 100 1 1
                
                gen_labels = (torch.rand(batch_size, 1) * N_Class).to(device) # 64 1
                gen_labels = gen_labels.long()
                gen_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                gen_y = gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).view(batch_size, N_Class,1,1) # 64 10 1 1
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # Loss for real images
                real_pred, real_aux = discriminator(real_imgs, real_y)

                print(real_pred.shape, real_aux.shape)
                print(valid.shape, labels.shape)
                d_real_loss = (self.adversarial_loss(real_pred.squeeze(), valid)*5 + self.auxiliary_loss(real_aux, labels)*5) / 10
                # Loss for fake images
                gen_imgs = generator(noise, gen_y) # 64 1 32 32
                gen_y_for_D = gen_y.view(batch_size, N_Class, 1, 1).contiguous().expand(-1, -1, img_size, img_size) # 64 10 32 32


                fake_pred, fake_aux = discriminator(gen_imgs.detach(),gen_y_for_D)
                #print(fake_aux.shape)
                #print(gen_labels.shape)
                d_fake_loss = (self.adversarial_loss(fake_pred.squeeze(), fake)*5 + self.auxiliary_loss(fake_aux, gen_labels.squeeze())*5) / 10

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss)/2
                d_loss.backward()
                optimizer_D.step()

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.squeeze().data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                validity, pred_label = discriminator(gen_imgs,gen_y_for_D)
                g_loss =  (self.adversarial_loss(validity.squeeze(), valid)*5 + self.auxiliary_loss(pred_label, gen_labels.squeeze())*5) / 10
                g_loss.backward()
                optimizer_G.step()


                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f , acc: %f%%] [G loss: %f]" % (epoch,self.args.pre_epochs, i, len(train_loader)-2,
                                                                    d_loss.data.cpu(), 100*d_acc, g_loss.data.cpu()))

        self.sample_cifarimage(self.args.num_classes, epoch + 1, generator)

    #生成cifar10
    def sample_cifarimage(self, n_row, batches_done, generator):
        """Saves a grid of generated digits ranging from 0 to num_classes"""
        # Sample noise
        noise = torch.rand((n_row ** 2, self.args.latent_dim,1,1)).to(device)
        y_ = torch.LongTensor(np.array([num for num in range(n_row)])).view(n_row,1).expand(-1,n_row).contiguous()
        y_ = y_.to(device)
        y_fixed = torch.zeros(n_row**2, n_row,device=device)
        y_fixed = y_fixed.scatter_(1,y_.view(n_row**2,1),1).view(n_row**2, n_row,1,1)

        gen_imgs = generator(noise, y_fixed).view(-1,self.args.channels,self.args.img_size,self.args.img_size)
        file_path = './generate/{}/{}_{}_{}clients_{}itss/'.format(self.args.dataset,self.args.epsilon,self.args.unsample_num,self.args.num_clients,self.args.num_its)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        save_image(gen_imgs.data, file_path+"%d.png" % batches_done, nrow=n_row, normalize=True)

    #BM_FL中单ITS上的训练cifar10
    def client_train_bmflcifar(self, generator, discriminator,optimizer_G, optimizer_D, discriminatorlast, trustedServer):
        epoch_loss=[]
        for wep in range(self.args.w_epochs):
            #获取当前客户端权重下的数据集
            hdataloader = self.getdataloader(trustedServer)
            for lep in range(self.args.train_ep):
                for i, (imgs, labels) in enumerate(hdataloader):
                    batch_size = self.args.train_bs  ## 64
                    N_Class = self.args.num_classes # 10
                    img_size = self.args.img_size  # 32
                    # Adversarial ground truths
                    valid = torch.FloatTensor(batch_size).fill_(1.0).to(device)
                    fake = torch.FloatTensor(batch_size).fill_(0.0).to(device)

                    # Configure input
                    real_imgs = imgs.to(device)  # 64 3 32 32
                    labels = labels.to(device)

                    if i == len(hdataloader)-2:
                        break
                    real_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                    real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, N_Class, 1, 1).contiguous() # 64 10 1 1
                    real_y = real_y.expand(-1, -1, img_size, img_size) # 64 10 32 32

                    # Sample noise and labels as generator input
                    noise = torch.randn(batch_size, self.args.latent_dim,1,1).to(device) # 64 100 1 1
                    
                    gen_labels = (torch.rand(batch_size, 1) * N_Class).to(device) # 64 1
                    gen_labels = gen_labels.long()
                    gen_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                    gen_y = gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).view(batch_size, N_Class,1,1) # 64 10 1 1
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    # Loss for real images
                    real_pred, real_aux = discriminator(real_imgs, real_y)

                    print(real_pred.shape, real_aux.shape)
                    print(valid.shape, labels.shape)
                    d_real_loss = (self.adversarial_loss(real_pred.squeeze(), valid)*5 + self.auxiliary_loss(real_aux, labels)*5) / 10
                    # Loss for fake images
                    gen_imgs = generator(noise, gen_y) # 64 1 32 32
                    gen_y_for_D = gen_y.view(batch_size, N_Class, 1, 1).contiguous().expand(-1, -1, img_size, img_size) # 64 10 32 32


                    fake_pred, fake_aux = discriminator(gen_imgs.detach(),gen_y_for_D)
                    #print(fake_aux.shape)
                    #print(gen_labels.shape)
                    d_fake_loss = (self.adversarial_loss(fake_pred.squeeze(), fake)*5 + self.auxiliary_loss(fake_aux, gen_labels.squeeze())*5) / 10

                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss)/2
                    d_loss.backward()
                    optimizer_D.step()

                    # Calculate discriminator accuracy
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.squeeze().data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    validity, pred_label = discriminator(gen_imgs,gen_y_for_D)
                    g_loss =  (self.adversarial_loss(validity.squeeze(), valid)*5 + self.auxiliary_loss(pred_label, gen_labels.squeeze())*5) / 10
                    g_loss.backward()
                    optimizer_G.step()


                    print ("[epoch %d/%d] [WEpoch %d/%d] [LEpoch %d/%d] [Batch %d/%d] [D loss: %f , acc: %f%%] [G loss: %f]" % (trustedServer.local_index,self.args.epoch,wep,self.args.w_epochs,lep,self.args.train_ep, i, len(hdataloader)-2,
                                                                        d_loss.data.cpu(), 100*d_acc, g_loss.data.cpu()))
            # 使用上一次权重更新的判别器模型更新数据集权重
            closs = []
            avgloss = 0.0
            for j in range(len(self.cdataloader)):
                # 计算每个客户端数据集的平均损失以及整体的平均损失
                temploss = 0.0
                for i, (imgs, labels) in enumerate(self.cdataloader[j]):
                    with torch.no_grad():
                        batch_size = imgs.shape[0]
                        N_Class = self.args.num_classes
                        img_size = self.args.img_size
                        if i == len(self.cdataloader[j])-2:
                            break
                        # Configure input
                        real_imgs = imgs.to(device)
                        labels = labels.to(device)
                        real_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                        real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, N_Class, 1, 1).contiguous() # 64 10 1 1
                        real_y = real_y.expand(-1, -1, img_size, img_size) # 64 10 32 32

                        # Loss for real images

                        real_pred, real_aux = discriminatorlast(real_imgs, real_y)
                        d_real_loss = self.auxiliary_loss(real_aux, labels)
                        temploss = temploss + d_real_loss
                temploss = temploss / (len(self.cdataloader[j])-2)
                closs.append(temploss)
            avgloss = sum(closs)/len(closs)
            epoch_loss.append(avgloss)
            for j in range(len(self.cdataloader)):
                # 更新权重
                cwight = trustedServer.clients[j].weights[-1]
                cwight = cwight + (closs[j] - avgloss) * (trustedServer.local_index +1) * 0.3
                trustedServer.clients[j].weights.append(cwight)
                #滑动更新
                self.sliding_average_update(trustedServer.clients[j].weights,self.args.slidL)
                if trustedServer.clients[j].weights[-1] > self.args.wup:
                    trustedServer.clients[j].weights[-1] = self.args.wup
                if trustedServer.clients[j].weights[-1] < self.args.wlow:
                    trustedServer.clients[j].weights[-1] = self.args.wlow

            discriminatorlast.load_state_dict(discriminator.state_dict())
        trustedServer.local_index = trustedServer.local_index +1
        return sum(epoch_loss)/len(epoch_loss)
    
    #测试cifar10
    def testcifar10(self, global_net, test_loader):
        global_net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for index, (data, labels) in enumerate(test_loader):
                batch_size = data.shape[0]
                N_Class = self.args.num_classes
                img_size = self.args.img_size
                if index == len(test_loader)-2:
                    break
                # Configure input
                real_imgs = data.to(device)
                labels = labels.to(device)
                valid = torch.FloatTensor(batch_size).fill_(1.0).to(device)
                real_y = torch.zeros(batch_size, N_Class, device=device)  # 64 10
                real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, N_Class, 1, 1).contiguous() # 64 10 1 1
                real_y = real_y.expand(-1, -1, img_size, img_size) # 64 10 32 32

                # Loss for real images

                real_pred, real_aux = global_net(real_imgs, real_y)

                d_real_loss = (self.adversarial_loss(real_pred.squeeze(), valid) + self.auxiliary_loss(real_aux, labels)) / 2
                total_loss += d_real_loss.item()  # 累积损失
                pred = real_aux.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()  # 累积正确的样本数
                total_samples += batch_size  # 累积总样本数
        average_loss = total_loss / (len(test_loader)-2)
        accuracy = total_correct / total_samples
        return average_loss, accuracy