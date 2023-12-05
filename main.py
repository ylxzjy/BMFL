import random
import torch

from models.Clients import ClientUpdate
from models.Getdataset import GetDataSet
from utils.bm_fl import bm_fl
from utils.preprocessDataset import preprocessData
from utils.config import args_parser


if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    args = args_parser()
    preprocessData(args)
    FL = ClientUpdate(args)
    getdata = GetDataSet(args)

    if args.FL_name == 'fedavg':
        fedavg(args, FL, getdata)
    elif args.FL_name == 'bm-fl':
        bm_fl(args, FL, getdata)
