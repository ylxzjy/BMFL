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


def bm_flmnist(args, FL, getdata):
    acc_list = []
    train_loss_list = []
    test_loss_list = []
    print("----------fed----------")
    print("dataset：{}   num_its:{}   ".format(args.dataset, args.num_its))
    global_dnet, global_gnet, ITSs_gnet, ITSs_dnet, optimizerg, optimizerd = FL.get_acmnistmodel()

    # load dataset：
    client_loader = []
    test_loader = None
    
    client_loader = getdata.NICO_train_loader
    test_loader = getdata.test_loader
    print(len(client_loader))
    ITSs_lastdnet = []

    # create client
    clients = []
    for idx in range(args.num_clients):
        client = Client(idx,client_loader[idx])
        clients.append(client)
    print("client create finished")
    # create ITS
    itss = []
    for idx in range(args.num_its):
        its_clients = [obj for idj, obj in enumerate(clients) if idj % args.num_its == idx]
        its = TrustedServer(its_clients)
        itss.append(its)
    print("itss create finished")
    del clients 
    # pretrain in ITS
    for idx in range(args.num_its):
         print("num of args.num_its :  ", idx)
         loss = FL.client_train_preacgan_mnist(ITSs_gnet[idx], ITSs_dnet[idx],optimizerg[idx], optimizerd[idx], client_loader[idx])

    print("pretrain finished")

    for idx in range(args.num_its):
        ITSs_lastdnet.append(ITSs_dnet[idx])

    # bm-fl
    for i in range(args.epoch):
        train_loss = 0
        for idx in range(args.num_its):
            loss = FL.client_train_bmfl(ITSs_gnet[idx], ITSs_dnet[idx], optimizerg[idx], optimizerd[idx], ITSs_lastdnet[idx],itss[idx])
            train_loss += loss
        
        # Aggregation Downstream Discriminator
        FL.FedAvg(global_dnet, [ITSs_dnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_dnet, ITSs_dnet)

        #Aggregation Downstream Generator
        FL.FedAvg(global_gnet, [ITSs_gnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_gnet, ITSs_gnet)
        
        # Generate samples
        for idx in range(args.num_its):
            FL.sample_mnistimage(args.num_classes, i*1000+idx, ITSs_gnet[idx])

        # Aggregation generator generates samples
        FL.sample_mnistimage(args.num_classes, i*1000+args.num_its, global_gnet)

        # Test Aggregation Discriminator Model
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

    # load dataset
    client_loader = []
    test_loader = None
    
    client_loader = getdata.NICO_train_loader
    test_loader = getdata.test_loader
    print(len(client_loader))
    ITSs_lastdnet = []

    # create client
    clients = []
    for idx in range(args.num_clients):
        client = Client(idx,client_loader[idx])
        clients.append(client)
    print("client create finished")
    # create ITS
    itss = []
    for idx in range(args.num_its):
        its_clients = [obj for idj, obj in enumerate(clients) if idj % args.num_its == idx]
        its = TrustedServer(its_clients)
        itss.append(its)
    print("itss create finished")
    # pretrain in ITS
    # for idx in range(args.num_its):
    #     print("num of args.num_its :  ", idx)
    #     loss = FL.client_train_preacgan_cifar(ITSs_gnet[idx], ITSs_dnet[idx],optimizerg[idx], optimizerd[idx], client_loader[idx])

    print("pretrain finished")

    for idx in range(args.num_its):
        ITSs_lastdnet.append(ITSs_dnet[idx])

    # bm-fl
    for i in range(args.epoch):
        train_loss = 0
        for idx in range(args.num_its):
            loss = FL.client_train_bmflcifar(ITSs_gnet[idx], ITSs_dnet[idx], optimizerg[idx], optimizerd[idx], ITSs_lastdnet[idx],itss[idx])
            train_loss += loss
        
        FL.FedAvg(global_dnet, [ITSs_dnet[idx] for idx in range(args.num_its)])
        FL.updata_model(global_dnet, ITSs_dnet)
        
        #generate samples
        for idx in range(args.num_its):
            FL.sample_cifarimage(args.num_classes, i*1000+idx, ITSs_gnet[idx])

        #Test Aggregation Discriminator Model
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
