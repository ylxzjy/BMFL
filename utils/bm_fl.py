import os.path
import numpy as np
from models.Clients import *
import torch

def bm_fl(args, FL, getdata):
    if args.dataset =="mnist":
        bm_flmnist(args, FL, getdata)
    elif args.dataset =="8fashion":
        bm_flmnist(args, FL, getdata)
    elif args.dataset =="cifar10":
        #bm_flmnist(args, FL, getdata)
        bm_flcifar10(args, FL, getdata)

#8fashion的尺寸大小和mnist相同使用和mnist一样的模型
def bm_flmnist(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   num_its:{}   ".format(args.dataset, args.num_its))
    global_dnet, global_gnet, ITSs_gnet, ITSs_dnet, optimizerg, optimizerd = FL.get_acmnistmodel()

    #加载每个客户端持有的数据集：getdata入手
    client_loader = []
    test_loader = None
    
    client_loader = getdata.NICO_train_loader
    test_loader = getdata.test_loader
    print(len(client_loader))
    ITSs_lastdnet = []

    # 创建client
    clients = []
    for idx in range(args.num_clients):
        client = Client(idx,client_loader[idx])
        clients.append(client)
    print("client create finished")
    # 创建ITS
    itss = []
    for idx in range(args.num_its):
        its_clients = [obj for idj, obj in enumerate(clients) if idj % args.num_its == idx]
        its = TrustedServer(its_clients)
        itss.append(its)
    print("itss create finished")
    del clients
    # 预训练中间信任服务器
    # for idx in range(args.num_its):
    #     print("num of args.num_its :  ", idx)
    #     loss = FL.client_train_preacgan_mnist(ITSs_gnet[idx], ITSs_dnet[idx],optimizerg[idx], optimizerd[idx], client_loader[idx])

    print("pretrain finished")

    for idx in range(args.num_its):
        ITSs_lastdnet.append(ITSs_dnet[idx])

    # 联邦训练中间信任服务器和中心服务器
    for i in range(args.epoch):
        train_loss = 0
        for idx in range(args.num_its):
            loss = FL.client_train_bmfl(ITSs_gnet[idx], ITSs_dnet[idx], optimizerg[idx], optimizerd[idx], ITSs_lastdnet[idx],itss[idx])
            train_loss += loss
        
        #聚合下发判别器
        FL.FedAvg(global_dnet, [ITSs_dnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_dnet, ITSs_dnet)

        #聚合下发生成器
        FL.FedAvg(global_gnet, [ITSs_gnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_gnet, ITSs_gnet)
        
        #每个中间信任服务器各自生成样本
        for idx in range(args.num_its):
            FL.sample_mnistimage(args.num_classes, i*1000+idx, ITSs_gnet[idx])

        #聚合生成器生成样本
        FL.sample_mnistimage(args.num_classes, i*1000+args.num_its, global_gnet)

        #测试聚合的判别器模型
        test_loss, accuracy = FL.test(global_dnet, test_loader)
        train_loss /= args.num_its
        accuracy = float(accuracy)

        print("Round {:3d}, Testing accuracy:{:.4f}".format(i + 1, accuracy))
        print("Train_loss:{:.5f}, Test_loss:{:.5f}".format(train_loss, test_loss))
        print("-" * 100)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        acc_list.append(accuracy)
        torch.cuda.empty_cache()
    file_path = './log/{}/{}ep_{}uns_{}clients_{}itss/acc.txt'.format(args.dataset, args.epsilon, args.unsample_num, args.num_clients, args.num_its)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    accfile = open(file_path, 'w')
    for ac in acc_list:
        temp = str(ac)
        accfile.write(temp)
        accfile.write('\n')
    accfile.close()


def bm_flcifar10(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   num_its:{}   ".format(args.dataset, args.num_its))
    global_dnet, ITSs_gnet, ITSs_dnet, optimizerg, optimizerd = FL.get_accifar10model()

    #加载每个客户端持有的数据集：getdata入手
    client_loader = []
    test_loader = None
    
    client_loader = getdata.NICO_train_loader
    test_loader = getdata.test_loader
    print(len(client_loader))
    ITSs_lastdnet = []

    # 创建client
    clients = []
    for idx in range(args.num_clients):
        client = Client(idx,client_loader[idx])
        clients.append(client)
    print("client create finished")
    # 创建ITS
    itss = []
    for idx in range(args.num_its):
        its_clients = [obj for idj, obj in enumerate(clients) if idj % args.num_its == idx]
        its = TrustedServer(its_clients)
        itss.append(its)
    print("itss create finished")
    # 预训练中间信任服务器
    # for idx in range(args.num_its):
    #     print("num of args.num_its :  ", idx)
    #     loss = FL.client_train_preacgan_cifar(ITSs_gnet[idx], ITSs_dnet[idx],optimizerg[idx], optimizerd[idx], client_loader[idx])

    print("pretrain finished")

    for idx in range(args.num_its):
        ITSs_lastdnet.append(ITSs_dnet[idx])

    # 联邦训练中间信任服务器和中心服务器
    for i in range(args.epoch):
        train_loss = 0
        for idx in range(args.num_its):
            loss = FL.client_train_bmflcifar(ITSs_gnet[idx], ITSs_dnet[idx], optimizerg[idx], optimizerd[idx], ITSs_lastdnet[idx],itss[idx])
            train_loss += loss
        
        FL.FedAvg(global_dnet, [ITSs_dnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_dnet, ITSs_dnet)
        
        #每个中间信任服务器各自生成样本
        for idx in range(args.num_its):
            FL.sample_cifarimage(args.num_classes, i*1000+idx, ITSs_gnet[idx])

        #测试聚合的判别器模型
        test_loss, accuracy = FL.testcifar10(global_dnet, test_loader)
        train_loss /= args.num_its
        accuracy = float(accuracy)

        print("Round {:3d}, Testing accuracy:{:.4f}".format(i + 1, accuracy))
        print("Train_loss:{:.5f}, Test_loss:{:.5f}".format(train_loss, test_loss))
        print("-" * 100)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        acc_list.append(accuracy)

    file_path = './log/{}/{}ep_{}uns_{}clients_{}itss/acc.txt'.format(args.dataset, args.epsilon, args.unsample_num, args.num_clients, args.num_its)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    accfile = open(file_path, 'w')
    for ac in acc_list:
        temp = str(ac)
        accfile.write(temp)
        accfile.write('\n')
    accfile.close()
